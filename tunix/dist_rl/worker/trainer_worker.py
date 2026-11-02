# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TrainerWorker implementation for role-based isolation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import contextlib
import functools
import time
from typing import Any, ContextManager, TYPE_CHECKING, cast

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.train import abstract_trainer
from tunix.dist_rl.worker import abstract_worker
from tunix.rl import common as rl_common
from tunix.rl import packing

if TYPE_CHECKING:
  from opentelemetry import metrics as otel_metrics_types  # pylint: disable=g-bad-import-order,g-import-not-at-top

try:
  from opentelemetry import metrics as otel_metrics  # pylint: disable=g-import-not-at-top
except ImportError:
  otel_metrics = None

WorkerState = datatypes.WorkerState


class _TrainerWorkerOtelMetrics:
  """Encapsulates OpenTelemetry instruments and recording for TrainerWorker.

  Note on async JAX dispatch (TODO(b/579110728)): `fwd_bwd` and `update` record
  host-side execution/dispatch duration so microbatch pipelining is not blocked
  by host synchronization; queued device execution time surfaces in the next
  host-synchronizing call (such as `prepare_weight_sync` ->
  `tunix.trainer.weight_sync.duration{direction="send"}` or `per_token_logps`).
  """

  def __init__(
      self,
      worker_id: str,
      meter_provider: otel_metrics_types.MeterProvider | None = None,
      clock: Callable[[], float] = time.time,
  ):
    self._worker_id = worker_id
    self._clock = clock
    self.last_free_time_s: float | None = None
    if otel_metrics is None:
      self._enabled = False
      return
    self._enabled = True

    provider = meter_provider or otel_metrics.get_meter_provider()
    meter = provider.get_meter("tunix.trainer")
    self._step_duration = meter.create_histogram(
        "tunix.trainer.step.duration",
        unit="s",
        description=(
            "Host-side execution or dispatch duration of trainer worker"
            " operations by phase."
        ),
    )
    self._tokens_processed = meter.create_counter(
        "tunix.trainer.tokens.processed",
        unit="{token}",
        description=(
            "Total valid and padded tokens processed by the trainer worker (on"
            " unpacked RL payloads without segment_ids, loss-masked tokens"
            " count as padded until valid_tokens metadata is stamped)."
        ),
    )
    self._weight_sync_duration = meter.create_histogram(
        "tunix.trainer.weight_sync.duration",
        unit="s",
        description=(
            "Duration of trainer weight sync staging and release (staging"
            " includes waiting on queued device work)."
        ),
    )
    self._starvation_duration = meter.create_histogram(
        "tunix.trainer.starvation.duration",
        unit="s",
        description=(
            "Trainer idle or data-fetch starvation time before a microbatch."
        ),
    )

  def mark_free(self, *, arm: bool = False) -> None:
    """Updates the wall-clock timestamp when the trainer worker becomes free."""
    if arm or self.last_free_time_s is not None:
      self.last_free_time_s = self._clock()

  def record_starvation(self, req_metadata: Mapping[str, Any] | None) -> None:
    """Records trainer_idle and data_fetch_delay before a microbatch."""
    if not self._enabled or self.last_free_time_s is None:
      return
    now_wall_s = self._clock()
    idle_s = max(0.0, now_wall_s - self.last_free_time_s)
    self._starvation_duration.record(
        float(idle_s),
        attributes={"worker_id": self._worker_id, "kind": "trainer_idle"},
    )
    if req_metadata and req_metadata.get("mb_ready_time_s") is not None:
      try:
        mb_ready_s = float(req_metadata["mb_ready_time_s"])
      except (TypeError, ValueError):
        return
      fetch_delay_s = max(
          0.0, now_wall_s - max(self.last_free_time_s, mb_ready_s)
      )
      self._starvation_duration.record(
          float(fetch_delay_s),
          attributes={
              "worker_id": self._worker_id,
              "kind": "data_fetch_delay",
          },
      )

  def record_step_duration(self, phase: str, duration_s: float) -> None:
    if self._enabled:
      self._step_duration.record(
          float(duration_s),
          attributes={"worker_id": self._worker_id, "phase": phase},
      )

  def record_weight_sync_duration(
      self, direction: str, duration_s: float
  ) -> None:
    if self._enabled:
      self._weight_sync_duration.record(
          float(duration_s),
          attributes={"worker_id": self._worker_id, "direction": direction},
      )

  def record_payload_tokens(self, payload: datatypes.TrainerPayload) -> None:
    """Records valid and padded token counts for an RL or SFT payload."""
    if not self._enabled:
      return
    valid_tokens: int | None = None
    total_tokens: int | None = None
    if isinstance(payload, datatypes.RLTrainerPayload):
      prompt_size = int(np.asarray(payload.prompt_ids).size)
      completion_size = int(np.asarray(payload.completion_ids).size)
      total_tokens = prompt_size + completion_size
      if payload.segment_ids is not None:
        seg_ids = np.asarray(payload.segment_ids)
        if seg_ids.size == total_tokens:
          valid_tokens = int(np.sum(seg_ids > 0))
        else:
          valid_tokens = int(
              np.sum(np.asarray(payload.prompt_mask) > 0) + np.sum(seg_ids > 0)
          )
      else:
        # TODO(b/579110728): Read stamped non-padding `valid_tokens` from
        # `payload.metadata` once `PaddedBatchAssembler` populates it, so
        # loss-masked multi-turn observation tokens are not counted as padded.
        valid_tokens = int(
            np.sum(np.asarray(payload.prompt_mask) > 0)
            + np.sum(np.asarray(payload.completion_mask) > 0)
        )
    elif isinstance(payload, datatypes.SFTTrainerPayload):
      total_tokens = int(np.asarray(payload.token_ids).size)
      if payload.segment_ids is not None:
        valid_tokens = int(np.sum(np.asarray(payload.segment_ids) > 0))
      else:
        valid_tokens = int(np.sum(np.asarray(payload.token_mask) > 0))
    if valid_tokens is not None and total_tokens is not None:
      valid_tokens = min(max(0, valid_tokens), total_tokens)
      padded_tokens = total_tokens - valid_tokens
      self._tokens_processed.add(
          valid_tokens,
          attributes={"worker_id": self._worker_id, "token_type": "valid"},
      )
      self._tokens_processed.add(
          padded_tokens,
          attributes={"worker_id": self._worker_id, "token_type": "padded"},
      )


def _compute_per_token_logps(
    model: nnx.Module, *args: Any, **kwargs: Any
) -> jax.Array:
  """Computes per-token log-probabilities under live model weights.

  Args:
    model: The NNX Module instance to score with.
    *args: Positional arguments forwarded to
      `rl_common.compute_per_token_logps`.
    **kwargs: Keyword arguments forwarded to
      `rl_common.compute_per_token_logps`.

  Returns:
    A JAX Array containing the computed per-token log-probabilities.
  """
  graphdef, state = nnx.split(model)
  return rl_common.compute_per_token_logps(
      graphdef, state, *args, stop_gradient=True, **kwargs
  )


class TrainerWorker(abstract_worker.Worker):
  """Worker wrapper for a Trainer.

  The TrainerWorker owns an AbstractTrainer and orchestrates its initialization,
  compilation, and execution. It exposes the trainer's API so it can be called
  by an orchestrator (either locally or via RPC).
  """

  def __init__(
      self,
      trainer_factory: Callable[[], abstract_trainer.AbstractTrainer],
      *,
      worker_id: str = "trainer_worker",
      logps_chunk_size: int = 0,
      logps_micro_batch_size: int | None = None,
      execution_context: Any = None,
      otel_meter_provider: otel_metrics_types.MeterProvider | None = None,
  ):
    """Initializes the TrainerWorker.

    Args:
      trainer_factory: A callable that returns an instantiated AbstractTrainer.
      worker_id: Unique identifier for this worker.
      execution_context: Optional context manager or zero-arg callable returning
        a context manager (e.g., a JAX Mesh) to enter during trainer
        initialization and worker method execution.
      logps_chunk_size: Optionally chunk the vocab (final-logits) computation.
      logps_micro_batch_size: Row chunk size for `per_token_logps`. If `None`,
        the whole request is scored in one forward.
      otel_meter_provider: Optional OpenTelemetry MeterProvider override for
        testing; defaults to the global MeterProvider when None.
    """
    self._execution_context = execution_context
    self._trainer_factory = trainer_factory
    self._trainer: abstract_trainer.AbstractTrainer | None = None
    self._pending_loss_fn: tuple[Callable[..., Any], bool] | None = None
    self._pending_gen_model_input_fn: Callable[[Any], dict[str, Any]] | None = (
        None
    )
    self._logps_chunk_size = logps_chunk_size
    self._logps_micro_batch_size = logps_micro_batch_size
    self._is_running = False
    self._worker_id = worker_id
    self._state = WorkerState.PENDING
    self._last_error: str | None = None
    # Set when `stop()` failed to close the trainer; later stops re-raise it.
    self._stop_error: str | None = None
    self._otel = _TrainerWorkerOtelMetrics(
        worker_id=worker_id, meter_provider=otel_meter_provider
    )

  def _policy_version(self) -> int:
    trainer = self._trainer
    if trainer is None:
      return 0
    return int(getattr(trainer, "policy_version", 0))

  def _response(self, **metadata: Any) -> datatypes.Response:
    return datatypes.Response(
        metadata={
            "worker_id": self._worker_id,
            "state": self.state.value,
            "policy_version": self._policy_version(),
            **metadata,
        }
    )

  def _ensure_ready(self) -> abstract_trainer.AbstractTrainer:
    if self.state == WorkerState.PENDING:
      self.initialize()
    if self.state != WorkerState.READY or self._trainer is None:
      raise RuntimeError(f"TrainerWorker is not ready: {self.state.value}.")
    return self._trainer

  @abstract_worker.offload_to_thread
  def initialize(self) -> datatypes.Response:
    """Initializes the worker and the underlying trainer."""
    if self.state == WorkerState.READY:
      return self._response(initialized=True, ready=True)
    self.state = WorkerState.INITIALIZING
    try:
      if self._trainer is None:
        with self.execution_context():
          trainer = self._trainer_factory()
          if self._pending_loss_fn is not None:
            trainer.with_loss_fn(*self._pending_loss_fn)
            self._pending_loss_fn = None
          if self._pending_gen_model_input_fn is not None:
            trainer.with_gen_model_input_fn(
                self._pending_gen_model_input_fn
            )
            self._pending_gen_model_input_fn = None
          self._trainer = trainer
      self._last_error = None
      return self._response(initialized=True)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    finally:
      if self.state == WorkerState.INITIALIZING:
        self.state = WorkerState.READY

  @abstract_worker.offload_to_thread
  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    """Triggers JIT compilation using the provided dummy_data."""
    trainer = self._ensure_ready()
    self.state = WorkerState.COMPILING
    try:
      dummy_data = abstract_trainer.resolve_dummy_routed_experts(
          trainer, dummy_data
      )
      trainer.compile(dummy_data)
      return self._response(compiled=True)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    finally:
      if self.state == WorkerState.COMPILING:
        self.state = WorkerState.READY
      self._otel.mark_free()

  @abstract_worker.offload_to_thread
  def start(self) -> datatypes.Response:
    """Starts the worker's main loop."""
    self._ensure_ready()
    self._is_running = True
    return self._response(started=True)

  @abstract_worker.offload_to_thread
  def stop(self, *, abort: bool = False) -> datatypes.Response:
    """Gracefully stops the worker, closing the trainer at most once.

    If closing the trainer fails, the worker still ends up `STOPPED` so that
    teardown never re-runs a partially failed close. The failure is kept instead
    of dropped: it is recorded as the worker's last error, escalated to an abort
    for the `set_on_stop` callback, and re-raised by every later `stop()` call,
    so local node teardown still exits non-zero.

    Args:
      abort: Whether the stop is part of an abnormal cluster teardown; see
        `Worker.stop`.

    Returns:
      A response with `stopped=True` and the requested `abort` flag, plus
      `already_stopped=True` if the worker was already stopped.

    Raises:
      RuntimeError: If an earlier `stop()` failed to close the trainer.
    """
    try:
      if self.state == WorkerState.STOPPED:
        if self._stop_error is not None:
          raise RuntimeError(
              f"TrainerWorker {self._worker_id!r} failed to stop cleanly"
              f" earlier: {self._stop_error}"
          )
        return self._response(
            stopped=True, already_stopped=True, abort=bool(abort)
        )
      self._is_running = False
      if self.state == WorkerState.READY:
        self.state = WorkerState.DRAINING
      try:
        if self._trainer is not None:
          self._trainer.close()
      except Exception as exc:
        self._stop_error = self._last_error = str(exc)
        abort = True
        raise
      finally:
        self.state = WorkerState.STOPPED
      return self._response(stopped=True, abort=bool(abort))
    finally:
      self._notify_stopped(abort)

  def info(self) -> datatypes.WorkerInfo:
    trainer = self._trainer
    return datatypes.WorkerInfo(
        worker_id=self._worker_id,
        roles=frozenset({"trainer", "weight_sync"}),
        resources={
            "trainer": (
                type(trainer).__name__
                if trainer is not None
                else "UninitializedTrainer"
            ),
            "policy_version": self._policy_version(),
        },
    )

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state=self.state,
        policy_version=self._policy_version(),
        last_error=self._last_error,
    )

  @abstract_worker.offload_to_thread
  def with_loss_fn(
      self, loss_fn: Callable[..., Any], has_aux: bool = False
  ) -> datatypes.Response:
    """Sets the loss function used by `fwd_bwd` (and evaluation).

    When the worker was built with a positive `logps_chunk_size`, it is bound
    into the loss as `compute_logps_chunk_size` (as the agentic GRPO learner
    does), so the training loss chunks the final-logits computation the same
    way `per_token_logps` does. A value supplied per call by
    `gen_model_input_fn` still takes precedence.

    Args:
      loss_fn: The loss function; must accept `compute_logps_chunk_size` as a
        keyword argument (or `**kwargs`) when chunking is enabled.
      has_aux: Whether `loss_fn` returns auxiliary outputs.

    Returns:
      A response confirming the loss function was configured.
    """
    if self._logps_chunk_size and self._logps_chunk_size > 0:
      loss_fn = functools.partial(
          loss_fn, compute_logps_chunk_size=self._logps_chunk_size
      )
    if self._trainer is None:
      self._pending_loss_fn = (loss_fn, has_aux)
    else:
      self._trainer.with_loss_fn(loss_fn, has_aux)
    return self._response(loss_fn_configured=True)

  @abstract_worker.offload_to_thread
  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> datatypes.Response:
    """Sets the last-mile adapter mapping a payload to the loss fn's kwargs."""
    if self._trainer is None:
      self._pending_gen_model_input_fn = gen_model_input_fn
    else:
      self._trainer.with_gen_model_input_fn(gen_model_input_fn)
    return self._response(gen_model_input_fn_configured=True)

  @abstract_worker.offload_to_thread
  def set_target_state(self, target_state: Any) -> datatypes.Response:
    """Stores rollout target_state so trainer-side weight sync can convert."""
    self._ensure_ready().set_target_state(target_state)
    return self._response(target_state_configured=True)

  @abstract_worker.offload_to_thread
  def fwd_bwd(self, request: datatypes.TrainRequest) -> datatypes.Response:
    """Executes one forward/backward pass."""
    trainer = self._ensure_ready()
    req_metadata = dict(request.metadata) if request.metadata else {}
    self._otel.record_starvation(req_metadata)
    try:
      # TODO(b/579110728): `fwd_bwd` records host dispatch duration under JAX
      # async dispatch; queued device time surfaces in the next synchronizing
      # RPC (`prepare_weight_sync` or `per_token_logps`).
      t0 = time.perf_counter()
      trainer.fwd_bwd(request.payload)
      elapsed_s = time.perf_counter() - t0
      self._last_error = None
      resp = self._response(queued=True, **req_metadata)
      resp.request_id = request.request_id
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    finally:
      self._otel.mark_free(arm=True)
    self._otel.record_step_duration("fwd_bwd", elapsed_s)
    self._otel.record_payload_tokens(request.payload)
    return resp

  @abstract_worker.offload_to_thread
  def update(self) -> tuple[int, Any]:
    """Applies the accumulated (mean) gradients and returns `(train_step, metrics)`."""
    trainer = self._ensure_ready()
    try:
      t0 = time.perf_counter()
      train_step = trainer.update()
      elapsed_s = time.perf_counter() - t0
      metrics = trainer.get_metrics()
      self._last_error = None
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    finally:
      self._otel.mark_free()
    self._otel.record_step_duration("optimizer_step", elapsed_s)
    return train_step, metrics

  @abstract_worker.offload_to_thread
  def eval_step(self, request: datatypes.TrainRequest) -> datatypes.Response:
    """Executes one evaluation step on the given payload."""
    trainer = self._ensure_ready()
    req_metadata = dict(request.metadata) if request.metadata else {}
    try:
      t0 = time.perf_counter()
      trainer.eval_step(request.payload)
      elapsed_s = time.perf_counter() - t0
      self._last_error = None
      resp = self._response(evaluated=True, **req_metadata)
      resp.request_id = request.request_id
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    finally:
      self._otel.mark_free()
    self._otel.record_step_duration("eval_step", elapsed_s)
    return resp

  @abstract_worker.offload_to_thread
  def run_eval(self, eval_ds: Any) -> datatypes.Response:
    """Runs an explicit evaluation phase over eval micro-batches."""
    trainer = self._ensure_ready()
    if eval_ds is None:
      return self._response(evaluated=True, eval_batches=0)
    try:
      run_eval = getattr(trainer, "run_eval", None)
      if callable(run_eval):
        run_eval(eval_ds)
        self._last_error = None
        return self._response(evaluated=True)

      eval_context = getattr(trainer, "eval_context", None)
      context = (
          cast(ContextManager[Any], eval_context())
          if callable(eval_context)
          else contextlib.nullcontext()
      )
      eval_batches = 0
      with context:
        for payload in eval_ds:
          t0 = time.perf_counter()
          trainer.eval_step(payload)
          self._otel.record_step_duration("eval_step", time.perf_counter() - t0)
          eval_batches += 1
      self._last_error = None
      return self._response(evaluated=True, eval_batches=eval_batches)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    finally:
      self._otel.mark_free()

  @abstract_worker.offload_to_thread
  def per_token_logps(
      self, items: datatypes.LogprobsRequest, **kwargs: Any
  ) -> datatypes.LogprobsResponse:
    """Scores per-token log-probs for a padded request under live weights.

    Read-only: unlike the frozen reference scorer this uses the trainer's
    current (actor) parameters, but it must not mutate trainer state. Accepts a
    ``LogprobsRequest`` composed by the orchestrator; the actor path sets
    ``pad_id``/``eos_id`` (and any packing fields) explicitly.

    Args:
      items: The log-probabilities request payload containing token sequences.
      **kwargs: Unused keyword arguments accepted for forwarding parity.

    Returns:
      A LogprobsResponse containing per-token log-probabilities and model
      version.
    """
    del kwargs  # Accepted for engine-forwarding parity; unused.
    trainer = self._ensure_ready()
    if items.pad_id is None or items.eos_id is None:
      raise ValueError(
          "TrainerWorker.per_token_logps requires pad_id and eos_id to be set "
          "on the LogprobsRequest; the actor scoring path must compose them."
      )
    try:
      prompt = np.asarray(items.prompt_tokens, dtype=np.int32)
      completion = np.asarray(items.completion_tokens, dtype=np.int32)
      batch_size = prompt.shape[0]
      if batch_size == 0:
        raise ValueError("per_token_logps requires a non-empty batch.")
      temperature = (
          1.0 if items.temperature is None else float(items.temperature)
      )
      seg_ids = (
          None
          if items.segment_ids is None
          else np.asarray(items.segment_ids, dtype=np.int32)
      )
      seg_pos = (
          None
          if items.segment_positions is None
          else np.asarray(items.segment_positions, dtype=np.int32)
      )
      if items.routed_experts is None:
        routed = None
      else:
        routed_raw = np.asarray(items.routed_experts)
        routed_enc = packing.routed_experts_encoding(routed_raw, batched=True)
        routed = (
            routed_raw
            if routed_enc in ("uint8", "uint10")
            else np.asarray(routed_raw, dtype=np.int16)
        )
      t0 = time.perf_counter()
      micro_batch_size = self._logps_micro_batch_size or batch_size
      outs = []
      for start in range(0, batch_size, micro_batch_size):
        sl = slice(start, start + micro_batch_size)
        outs.append(
            trainer.fwd_only(
                _compute_per_token_logps,
                prompt[sl],
                completion[sl],
                pad_id=items.pad_id,
                eos_id=items.eos_id,
                temperature=temperature,
                chunk_size=self._logps_chunk_size,
                segment_ids=None if seg_ids is None else seg_ids[sl],
                segment_positions=None if seg_pos is None else seg_pos[sl],
                routed_experts=None if routed is None else routed[sl],
            )
        )
      result = np.asarray(jnp.concatenate(outs, axis=0), dtype=np.float32)
      elapsed_s = time.perf_counter() - t0
      self._last_error = None
      resp = datatypes.LogprobsResponse(
          request_id=items.request_id,
          per_token_logps=result,
          model_version=self._policy_version(),
      )
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    finally:
      self._otel.mark_free()
    self._otel.record_step_duration("per_token_logps", elapsed_s)
    return resp

  @abstract_worker.offload_to_thread
  def save_checkpoint(
      self,
      metadata: Any = None,
      *,
      step: int | None = None,
      force: bool = False,
  ) -> datatypes.Response:
    """Force the trainer to serialize its state (model + optimizer)."""
    trainer = self._ensure_ready()
    try:
      t0 = time.perf_counter()
      trainer.save_checkpoint(metadata, step=step, force=force)
      elapsed_s = time.perf_counter() - t0
      self._last_error = None
      resp = self._response(checkpoint_saved=True)
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    finally:
      self._otel.mark_free()
    self._otel.record_step_duration("checkpoint_save", elapsed_s)
    return resp

  @abstract_worker.offload_to_thread
  def restore_checkpoint(self, step: int | None = None) -> Any:
    """Restore state from a checkpoint and return the metadata pytree."""
    try:
      return self._ensure_ready().restore_checkpoint(step=step)
    finally:
      self._otel.mark_free()

  @abstract_worker.offload_to_thread
  def prepare_weight_sync(self, sync_request: Any = None) -> Any:
    """Stages weights for transfer and returns their metadata."""
    trainer = self._ensure_ready()
    self.state = WorkerState.SYNCING
    try:
      t0 = time.perf_counter()
      metadata = trainer.prepare_weight_sync(sync_request=sync_request)
      elapsed_s = time.perf_counter() - t0
      self._last_error = None
      resp = (
          metadata
          if metadata is not None
          else self._response(weight_sync_ready=True)
      )
    except Exception as exc:
      self._last_error = str(exc)
      self.state = WorkerState.ERROR
      raise
    finally:
      self._otel.mark_free()
    self._otel.record_weight_sync_duration("send", elapsed_s)
    return resp

  @abstract_worker.offload_to_thread
  def release_weight_sync(self, sync_request: Any = None) -> Any:
    """Releases this round's staging and restores READY."""
    # Not `_ensure_ready()`: `prepare_weight_sync` left the state at SYNCING,
    # and `_ensure_ready()` raises unless the state is READY.
    was_initialized = self._trainer is not None
    t0 = time.perf_counter()
    try:
      result = (
          self._trainer.release_weight_sync(sync_request=sync_request)
          if self._trainer is not None
          else None
      )
    finally:
      self._otel.mark_free()
    if was_initialized:
      self._otel.record_weight_sync_duration(
          "release", time.perf_counter() - t0
      )
    if self.state == WorkerState.SYNCING:
      self.state = WorkerState.READY
    return result

  @abstract_worker.offload_to_thread
  def get_metrics(self) -> Any:
    """Returns and clears the recently collected step metric records."""
    return self._ensure_ready().get_metrics()
