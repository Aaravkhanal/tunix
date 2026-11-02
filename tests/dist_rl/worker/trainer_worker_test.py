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

"""Unit tests for TrainerWorker.

Verifies TrainerWorker RPC request unpacking (TrainRequest), delegation to
AbstractTrainer (fwd_bwd, eval_step, update), response metadata stamping, and
per-token log-prob scorer run through AbstractTrainer.fwd_only.
"""

import collections
import dataclasses
import functools
import threading
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
from opentelemetry.sdk import metrics as sdk_metrics
from opentelemetry.sdk.metrics import export as sdk_metrics_export
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.train import abstract_trainer
from tunix.dist_rl.worker import trainer_worker
from tunix.rl import algo_core
from tunix.rl import algorithm_config
from tunix.rl import common as rl_common
from tunix.tests import test_common as tc


class FakeTrainer(abstract_trainer.AbstractTrainer):

  def __init__(self):
    self.compile_calls = []
    self.fwd_bwd_calls = []
    self.eval_step_calls = []
    self.fwd_only_calls = []
    self.model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    self.policy_version = 3
    self.step_count = 10
    self.target_state = None
    self.gen_model_input_fn = None

  def compile(self, dummy_data=None):
    self.compile_calls.append(dummy_data)

  def with_loss_fn(self, loss_fn, has_aux=False):
    self.loss_fn = loss_fn
    self.has_aux = has_aux

  def with_gen_model_input_fn(self, gen_model_input_fn):
    self.gen_model_input_fn = gen_model_input_fn

  def fwd_bwd(self, payload):
    self.fwd_bwd_calls.append(payload)

  def update(self):
    self.step_count += 1
    return self.step_count

  def eval_step(self, payload):
    self.eval_step_calls.append(payload)

  def fwd_only(self, fn, *args, **kwargs):
    self.fwd_only_calls.append((args, kwargs))
    return fn(self.model, *args, **kwargs)

  def save_checkpoint(self, metadata=None, *, step=None, force=False):
    pass

  def restore_checkpoint(self, *, step=None):
    return {}

  def get_metrics(self):
    return {"loss": 0.25}

  def prepare_weight_sync(self, *, sync_request=None):
    pass

  def set_target_state(self, target_state: Any) -> None:
    self.target_state = target_state

  def close(self):
    pass


class TrainerWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.fake_trainer = FakeTrainer()
    self.worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_0",
    )
    self.worker.initialize()

  def test_trainer_factory_deferred_until_initialize(self):
    calls = []

    def _factory():
      calls.append("called")
      return self.fake_trainer

    worker = trainer_worker.TrainerWorker(
        trainer_factory=_factory, worker_id="lazy_trainer"
    )
    gen_fn = lambda x: {"input": x}
    worker.with_gen_model_input_fn(gen_fn)
    self.assertEmpty(calls)
    self.assertEqual(worker.state, datatypes.WorkerState.PENDING)
    self.assertEqual(worker.heartbeat().policy_version, 0)
    self.assertEqual(worker.info().resources["trainer"], "UninitializedTrainer")

    worker.initialize()
    self.assertEqual(calls, ["called"])
    self.assertEqual(worker.state, datatypes.WorkerState.READY)
    self.assertEqual(worker.heartbeat().policy_version, 3)
    self.assertEqual(worker.info().resources["trainer"], "FakeTrainer")
    self.assertIs(self.fake_trainer.gen_model_input_fn, gen_fn)

  def test_initialize_failure_sets_error_state(self):
    def _failing_factory():
      raise RuntimeError("weight load failed")

    worker = trainer_worker.TrainerWorker(
        trainer_factory=_failing_factory, worker_id="bad_trainer"
    )
    with self.assertRaisesRegex(RuntimeError, "weight load failed"):
      worker.initialize()
    self.assertEqual(worker.state, datatypes.WorkerState.ERROR)
    self.assertEqual(worker.heartbeat().last_error, "weight load failed")

    # Verify a failure in a pending hook leaves _trainer as None so a retry
    # re-runs initialization cleanly.
    should_fail = True
    orig_with_loss_fn = self.fake_trainer.with_loss_fn

    def _flaky_with_loss_fn(loss_fn, has_aux=False):
      if should_fail:
        raise RuntimeError("loss hook failed")
      orig_with_loss_fn(loss_fn, has_aux)

    self.fake_trainer.with_loss_fn = _flaky_with_loss_fn
    retry_worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer, worker_id="retry_trainer"
    )
    retry_worker.with_loss_fn(lambda m, **kw: kw)
    with self.assertRaisesRegex(RuntimeError, "loss hook failed"):
      retry_worker.initialize()
    self.assertIsNone(retry_worker._trainer)
    self.assertEqual(retry_worker.state, datatypes.WorkerState.ERROR)

    should_fail = False
    retry_worker._state = datatypes.WorkerState.PENDING
    retry_worker.initialize()
    self.assertIs(retry_worker._trainer, self.fake_trainer)
    self.assertEqual(retry_worker.state, datatypes.WorkerState.READY)
    self.assertIsNotNone(self.fake_trainer.loss_fn)

  def test_with_loss_fn_binds_logps_chunk_size(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer, logps_chunk_size=2048
    )
    worker.with_loss_fn(lambda model, **kwargs: kwargs, has_aux=True)
    worker.initialize()

    self.assertTrue(self.fake_trainer.has_aux)
    self.assertEqual(
        self.fake_trainer.loss_fn(None, pad_id=0),
        {"pad_id": 0, "compute_logps_chunk_size": 2048},
    )

  def test_with_loss_fn_per_call_chunk_size_wins(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer, logps_chunk_size=2048
    )
    worker.with_loss_fn(lambda model, **kwargs: kwargs)
    worker.initialize()

    kwargs = self.fake_trainer.loss_fn(None, compute_logps_chunk_size=512)

    self.assertEqual(kwargs["compute_logps_chunk_size"], 512)

  def test_with_loss_fn_unchanged_without_chunk_size(self):
    fn = lambda model, **kwargs: kwargs
    self.worker.with_loss_fn(fn)

    self.assertIs(self.fake_trainer.loss_fn, fn)

  def test_compile_forwards_dummy_data_to_trainer(self):
    dummy = {"dummy": 1}
    resp = self.worker.compile(dummy)
    self.assertTrue(resp.metadata["compiled"])
    self.assertEqual(self.worker.state, datatypes.WorkerState.READY)
    self.assertEqual(self.fake_trainer.compile_calls, [dummy])

  def test_compile_resolves_dummy_routed_experts_for_maxtext_engine(self):
    encode_fn = functools.partial(
        trainer_worker.packing.encode_routed_experts,
        mode="auto",
        batched=True,
    )

    class _MaxTextCfg:
      num_decoder_layers = 40
      num_experts_per_tok = 8
      num_experts = 256

    class _BaseModel:
      config = _MaxTextCfg()

    class _AdapterModel:
      # Mirrors `maxtext.utils.model_creation_utils.from_pretrained(...,
      # wrap_with_tunix_adapter=True)`, which explicitly sets `model.config =
      # None` on `TunixMaxTextAdapter`.
      config = None
      base = _BaseModel()

    # 1. Public `trainer.model.base.config` (TunixMaxTextAdapter) with real
    # `packing.encode_routed_experts` -> 5D uint10 `(2, 8, 40, 2, 5)`.
    self.fake_trainer.model = _AdapterModel()
    dummy = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros((2, 4), dtype=np.int32),
        prompt_mask=np.ones((2, 4), dtype=np.int32),
        completion_ids=np.zeros((2, 4), dtype=np.int32),
        completion_mask=np.ones((2, 4), dtype=np.int32),
        advantages=np.zeros((2, 8), dtype=np.float32),
        routed_experts=np.zeros((2, 8, 0, 0), dtype=np.int16),
        metadata={"routed_experts_encode_fn": encode_fn, "tag": "moe"},
    )
    resp = self.worker.compile(dummy)
    self.assertTrue(resp.metadata["compiled"])
    self.assertLen(self.fake_trainer.compile_calls, 1)
    compiled_payload = self.fake_trainer.compile_calls[0]
    self.assertEqual(compiled_payload.routed_experts.shape, (2, 8, 40, 2, 5))
    self.assertEqual(compiled_payload.routed_experts.dtype, np.uint8)
    self.assertEqual(compiled_payload.metadata, {"tag": "moe"})

    # 2. No `routed_experts_encode_fn` -> raw `(2, 8, 40, 8)` int16 with
    # metadata preserved unchanged.
    self.fake_trainer.compile_calls.clear()
    raw_dummy = dataclasses.replace(dummy, metadata={"keep": 1})
    self.worker.compile(raw_dummy)
    raw_compiled = self.fake_trainer.compile_calls[0]
    self.assertEqual(raw_compiled.routed_experts.shape, (2, 8, 40, 8))
    self.assertEqual(raw_compiled.routed_experts.dtype, np.int16)
    self.assertEqual(raw_compiled.metadata, {"keep": 1})

    # 3. Native `trainer.model.config.num_layers` path (`num_experts <= 255`)
    # -> 4D uint8 `(2, 8, 12, 4)`.
    class _NativeCfg:
      num_layers = 12
      num_experts_per_tok = 4
      num_experts = 64

    class _NativeModel:
      config = _NativeCfg()

    self.fake_trainer.model = _NativeModel()
    self.fake_trainer.compile_calls.clear()
    self.worker.compile(dummy)
    native_compiled = self.fake_trainer.compile_calls[0]
    self.assertEqual(native_compiled.routed_experts.shape, (2, 8, 12, 4))
    self.assertEqual(native_compiled.routed_experts.dtype, np.uint8)
    self.assertEqual(native_compiled.metadata, {"tag": "moe"})

    # 4. Non-empty `routed_experts` is forwarded untouched.
    self.fake_trainer.compile_calls.clear()
    non_empty = dataclasses.replace(
        dummy,
        routed_experts=np.ones((2, 8, 12, 4), dtype=np.int16),
        metadata={"keep": 2},
    )
    self.worker.compile(non_empty)
    passthrough = self.fake_trainer.compile_calls[0]
    self.assertIs(passthrough, non_empty)
    self.assertIs(
        abstract_trainer.resolve_dummy_routed_experts(
            self.fake_trainer, compiled_payload
        ),
        compiled_payload,
    )

    # 5. Malformed rank < 2 zero-sized placeholder fails loud.
    malformed = dataclasses.replace(
        dummy, routed_experts=np.zeros((0,), dtype=np.int16)
    )
    with self.assertRaisesRegex(ValueError, "rank >= 2"):
      abstract_trainer.resolve_dummy_routed_experts(
          self.fake_trainer, malformed
      )

    # 6. Model without any non-None config fails loud with a clear ValueError.
    class _ConfiglessModel:
      config = None

    self.fake_trainer.model = _ConfiglessModel()
    with self.assertRaisesRegex(ValueError, "non-None"):
      abstract_trainer.resolve_dummy_routed_experts(self.fake_trainer, dummy)

  def test_offload_markers_and_stop_serializes_with_inflight_step(self):
    attr = "_tunix_offload_to_thread"
    for offloaded_name in (
        "initialize",
        "compile",
        "start",
        "stop",
        "with_loss_fn",
        "with_gen_model_input_fn",
        "set_target_state",
        "fwd_bwd",
        "update",
        "eval_step",
        "run_eval",
        "per_token_logps",
        "save_checkpoint",
        "restore_checkpoint",
        "prepare_weight_sync",
        "release_weight_sync",
        "get_metrics",
    ):
      self.assertTrue(
          getattr(getattr(self.worker, offloaded_name), attr, False),
          msg=f"{offloaded_name} should opt in to worker-thread offloading",
      )
    for inline_name in ("heartbeat", "info"):
      self.assertFalse(
          getattr(getattr(self.worker, inline_name), attr, False),
          msg=f"{inline_name} must remain inline on the event loop",
      )

    entered_step = threading.Event()
    release_step = threading.Event()
    close_called = threading.Event()

    def _blocking_fwd_bwd(payload):
      del payload
      entered_step.set()
      if not release_step.wait(timeout=10.0):
        raise TimeoutError("release_step timed out")

    def _recording_close():
      close_called.set()

    self.fake_trainer.fwd_bwd = _blocking_fwd_bwd
    self.fake_trainer.close = _recording_close
    req = datatypes.TrainRequest(
        request_id="req-lock",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[1]], dtype=np.int32),
            prompt_mask=np.ones((1, 1), dtype=np.float32),
            completion_ids=np.array([[2]], dtype=np.int32),
            completion_mask=np.array([[1]], dtype=np.float32),
            advantages=np.array([1.0], dtype=np.float32),
        ),
    )
    step_thread = threading.Thread(
        target=lambda: self.worker.fwd_bwd(request=req)
    )
    stop_thread = threading.Thread(target=self.worker.stop)
    step_thread.start()
    self.assertTrue(entered_step.wait(timeout=5.0))
    # heartbeat() and info() must remain non-blocking while _trainer_lock is
    # held by an in-flight fwd_bwd step.
    self.assertEqual(self.worker.heartbeat().state, datatypes.WorkerState.READY)
    self.assertEqual(self.worker.info().worker_id, "trainer_0")

    stop_thread.start()
    self.assertFalse(close_called.wait(timeout=0.05))
    release_step.set()
    step_thread.join(timeout=5.0)
    stop_thread.join(timeout=5.0)
    self.assertFalse(step_thread.is_alive())
    self.assertFalse(stop_thread.is_alive())
    self.assertTrue(close_called.is_set())
    self.assertEqual(self.worker.state, datatypes.WorkerState.STOPPED)

  def test_with_loss_fn_chunked_grpo_loss_matches_unchunked(self):
    batch, prompt_len, completion_len = 2, 3, 4
    example = datatypes.RLTrainerPayload(
        prompt_ids=np.arange(1, 1 + batch * prompt_len, dtype=np.int32).reshape(
            batch, prompt_len
        ),
        prompt_mask=np.ones((batch, prompt_len), dtype=np.int32),
        completion_ids=np.arange(
            1, 1 + batch * completion_len, dtype=np.int32
        ).reshape(batch, completion_len),
        completion_mask=np.ones((batch, completion_len), dtype=np.int32),
        advantages=np.array([1.0, -1.0], dtype=np.float32),
    )
    algo_config = algorithm_config.GRPOConfig(beta=0.0)
    algo_config.temperature = 1.0

    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer, logps_chunk_size=3
    )
    worker.with_loss_fn(algo_core.grpo_loss_fn)
    worker.initialize()
    chunked_fn = self.fake_trainer.loss_fn
    self.assertEqual(chunked_fn.keywords, {"compute_logps_chunk_size": 3})

    kwargs = dict(
        train_example=example, algo_config=algo_config, pad_id=0, eos_id=0
    )
    chunked = chunked_fn(self.fake_trainer.model, **kwargs)
    unchunked = algo_core.grpo_loss_fn(self.fake_trainer.model, **kwargs)

    np.testing.assert_allclose(
        np.asarray(chunked.primary_loss.compute()),
        np.asarray(unchunked.primary_loss.compute()),
        rtol=1e-5,
        atol=1e-5,
    )

  def test_fwd_bwd_with_train_request(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2], [1, 2]], dtype=np.int32),
        prompt_mask=np.ones((2, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4], [3, 4]], dtype=np.int32),
        completion_mask=np.array([[1, 1], [1, 0]], dtype=np.float32),
        advantages=np.array([1.0, 2.0], dtype=np.float32),
        metadata={"step": 1},
    )
    request = datatypes.TrainRequest(
        request_id="req-train-123",
        payload=payload,
        metadata={"batch_id": "b0"},
    )

    resp = self.worker.fwd_bwd(request=request)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertEqual(resp.request_id, "req-train-123")
    self.assertEqual(resp.metadata["worker_id"], "trainer_0")
    self.assertEqual(resp.metadata["batch_id"], "b0")
    self.assertEqual(resp.metadata["policy_version"], 3)
    self.assertTrue(resp.metadata["queued"])
    self.assertLen(self.fake_trainer.fwd_bwd_calls, 1)
    self.assertIs(self.fake_trainer.fwd_bwd_calls[0], payload)

  def test_eval_step_with_train_request(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1]], dtype=np.int32),
        prompt_mask=np.ones((1, 1), dtype=np.float32),
        completion_ids=np.array([[2]], dtype=np.int32),
        completion_mask=np.array([[1]], dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    request = datatypes.TrainRequest(
        request_id="req-eval-456",
        payload=payload,
        metadata={"eval_split": "val"},
    )

    resp = self.worker.eval_step(request=request)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertEqual(resp.request_id, "req-eval-456")
    self.assertEqual(resp.metadata["eval_split"], "val")
    self.assertTrue(resp.metadata["evaluated"])
    self.assertLen(self.fake_trainer.eval_step_calls, 1)
    self.assertIs(self.fake_trainer.eval_step_calls[0], payload)

  def test_update_returns_step_count(self):
    step, metrics = self.worker.update()
    self.assertEqual(step, 11)
    self.assertEqual(metrics, {"loss": 0.25})

  def test_set_target_state_configures_trainer(self):
    target_state = {"params": np.zeros((4, 4))}
    resp = self.worker.set_target_state(target_state=target_state)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertTrue(resp.metadata["target_state_configured"])
    self.assertEqual(self.fake_trainer.target_state, target_state)

  def test_save_checkpoint_metadata_is_optional(self):
    with mock.patch.object(
        self.fake_trainer, "save_checkpoint", autospec=True
    ) as save:
      self.worker.save_checkpoint(step=42, force=True)
    save.assert_called_once_with(None, step=42, force=True)

  def test_set_target_state_falls_back_to_abstract_default(self):
    with mock.patch.object(
        FakeTrainer,
        "set_target_state",
        abstract_trainer.AbstractTrainer.set_target_state,
    ):
      resp = self.worker.set_target_state(target_state={"params": 1})
    self.assertTrue(resp.metadata["target_state_configured"])
    self.assertIsNone(self.fake_trainer.target_state)

  def _expected_logps(self, request, chunk_size=0):
    graphdef, state = nnx.split(self.fake_trainer.model)
    return np.asarray(
        rl_common.compute_per_token_logps(
            graphdef,
            state,
            prompt_tokens=jnp.asarray(request.prompt_tokens),
            completion_tokens=jnp.asarray(request.completion_tokens),
            pad_id=request.pad_id,
            eos_id=request.eos_id,
            stop_gradient=True,
            temperature=request.temperature,
            chunk_size=chunk_size,
            segment_ids=(
                None
                if request.segment_ids is None
                else jnp.asarray(request.segment_ids)
            ),
            segment_positions=(
                None
                if request.segment_positions is None
                else jnp.asarray(request.segment_positions)
            ),
        )
    )

  def _unpacked_request(self, temperature=1.0):
    batch, prompt_len, completion_len = 3, 3, 4
    return datatypes.LogprobsRequest(
        prompt_tokens=np.arange(
            1, 1 + batch * prompt_len, dtype=np.int32
        ).reshape(batch, prompt_len),
        completion_tokens=np.arange(
            1, 1 + batch * completion_len, dtype=np.int32
        ).reshape(batch, completion_len),
        temperature=temperature,
        pad_id=0,
        eos_id=0,
    )

  def test_per_token_logps_matches_reference_scorer(self):
    request = self._unpacked_request()

    result = self.worker.per_token_logps(items=request)

    self.assertIsInstance(result, datatypes.LogprobsResponse)
    self.assertEqual(result.request_id, request.request_id)
    self.assertEqual(result.model_version, self.fake_trainer.policy_version)
    self.assertIsInstance(result.per_token_logps, np.ndarray)
    self.assertEqual(result.per_token_logps.dtype, np.float32)
    self.assertEqual(result.per_token_logps.shape, (3, 4))
    # Whole request in one forward by default.
    self.assertLen(self.fake_trainer.fwd_only_calls, 1)
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_per_token_logps_micro_batches_in_order(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_1",
        logps_micro_batch_size=2,
    )
    worker.initialize()
    request = self._unpacked_request()

    result = worker.per_token_logps(items=request)

    self.assertLen(self.fake_trainer.fwd_only_calls, 2)
    first_args, _ = self.fake_trainer.fwd_only_calls[0]
    second_args, _ = self.fake_trainer.fwd_only_calls[1]
    np.testing.assert_array_equal(first_args[1], request.completion_tokens[:2])
    np.testing.assert_array_equal(second_args[1], request.completion_tokens[2:])
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_per_token_logps_temperature_and_chunked_vocab(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_2",
        logps_chunk_size=3,
    )
    worker.initialize()
    request = self._unpacked_request(temperature=0.7)

    result = worker.per_token_logps(items=request)

    _, kwargs = self.fake_trainer.fwd_only_calls[0]
    self.assertEqual(kwargs["temperature"], 0.7)
    self.assertEqual(kwargs["chunk_size"], 3)
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request, chunk_size=3),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_per_token_logps_packed_request(self):
    request = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((2, 0), dtype=np.int32),
        completion_tokens=np.array([[3, 4, 5], [3, 4, 0]], dtype=np.int32),
        temperature=1.0,
        pad_id=0,
        eos_id=0,
        segment_ids=np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int32),
        segment_positions=np.array([[0, 1, 2], [0, 1, 0]], dtype=np.int32),
    )

    result = self.worker.per_token_logps(items=request)

    self.assertEqual(result.per_token_logps.shape, (2, 3))
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_per_token_logps_empty_batch_raises(self):
    request = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((0, 3), dtype=np.int32),
        completion_tokens=np.zeros((0, 4), dtype=np.int32),
        temperature=1.0,
        pad_id=0,
        eos_id=0,
    )
    with self.assertRaises(ValueError):
      self.worker.per_token_logps(items=request)
    self.assertEmpty(self.fake_trainer.fwd_only_calls)
    self.assertEqual(self.worker.state, datatypes.WorkerState.ERROR)

  def test_per_token_logps_requires_pad_and_eos(self):
    request = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((1, 0), dtype=np.int32),
        completion_tokens=np.array([[3, 4]], dtype=np.int32),
        temperature=1.0,
    )
    with self.assertRaises(ValueError):
      self.worker.per_token_logps(items=request)
    self.assertEmpty(self.fake_trainer.fwd_only_calls)

  def test_per_token_logps_without_routing_passes_none(self):
    self.worker.per_token_logps(items=self._unpacked_request())
    _, kwargs = self.fake_trainer.fwd_only_calls[0]
    self.assertIsNone(kwargs["routed_experts"])

  def test_per_token_logps_forwards_routed_experts_per_micro_batch(self):
    worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_3",
        logps_micro_batch_size=2,
    )
    worker.initialize()
    request = self._unpacked_request()
    batch, seq_len = 3, 3 + 4
    # Row i routes every token to expert i, so a mis-sliced row is visible.
    routed = np.broadcast_to(
        np.arange(batch, dtype=np.int32).reshape(batch, 1, 1, 1),
        (batch, seq_len, 2, 2),
    )
    request = dataclasses.replace(request, routed_experts=routed)

    result = worker.per_token_logps(items=request)

    self.assertLen(self.fake_trainer.fwd_only_calls, 2)
    _, first_kwargs = self.fake_trainer.fwd_only_calls[0]
    _, second_kwargs = self.fake_trainer.fwd_only_calls[1]
    self.assertEqual(first_kwargs["routed_experts"].dtype, np.int16)
    np.testing.assert_array_equal(first_kwargs["routed_experts"], routed[:2])
    np.testing.assert_array_equal(second_kwargs["routed_experts"], routed[2:])
    # The fake model does not accept `forced_routed_experts`, so scoring is
    # unchanged; the routing is only forwarded.
    np.testing.assert_allclose(
        result.per_token_logps,
        self._expected_logps(request),
        rtol=1e-4,
        atol=1e-4,
    )

    # Also verify compact uint8 (4D) and uint10 (5D) routed_experts are kept
    # in compact uint8 form without widening to int16.
    for shape in ((batch, seq_len, 2, 4), (batch, seq_len, 2, 1, 5)):
      self.fake_trainer.fwd_only_calls.clear()
      compact_routed = np.ones(shape, dtype=np.uint8)
      compact_req = dataclasses.replace(request, routed_experts=compact_routed)
      worker.per_token_logps(items=compact_req)
      self.assertLen(self.fake_trainer.fwd_only_calls, 2)
      _, kw0 = self.fake_trainer.fwd_only_calls[0]
      _, kw1 = self.fake_trainer.fwd_only_calls[1]
      self.assertEqual(kw0["routed_experts"].dtype, np.uint8)
      self.assertEqual(kw0["routed_experts"].shape, (2,) + shape[1:])
      self.assertEqual(kw1["routed_experts"].dtype, np.uint8)
      self.assertEqual(kw1["routed_experts"].shape, (1,) + shape[1:])


class TrainerWorkerExecutionContextTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.events = []

    class TrackingContext:

      def __init__(self, events):
        self._events = events

      def __enter__(self):
        self._events.append("enter_ctx")
        return self

      def __exit__(self, *args):
        self._events.append("exit_ctx")

    self.ctx = TrackingContext(self.events)
    self.fake_trainer = FakeTrainer()

    def _factory():
      self.events.append("create_trainer")
      return self.fake_trainer

    self.worker = trainer_worker.TrainerWorker(
        trainer_factory=_factory,
        worker_id="trainer_ctx",
        execution_context=self.ctx,
    )

  def test_trainer_factory_runs_within_execution_context_on_initialize(self):
    self.assertEmpty(self.events)
    self.worker.initialize()
    self.assertEqual(self.events, ["enter_ctx", "create_trainer", "exit_ctx"])

  def test_all_worker_operations_run_within_execution_context(self):
    self.events.clear()
    self.worker.initialize()
    self.assertEqual(self.events, ["enter_ctx", "create_trainer", "exit_ctx"])

    # Verify per_token_logps runs inside execution_context.
    self.events.clear()
    logps_req = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((1, 0), dtype=np.int32),
        completion_tokens=np.array([[3, 4]], dtype=np.int32),
        temperature=1.0,
        pad_id=0,
        eos_id=1,
    )
    self.worker.per_token_logps(items=logps_req)
    self.assertEqual(self.events, ["enter_ctx", "exit_ctx"])

    # Verify fwd_bwd runs inside execution_context.
    self.events.clear()
    train_req = datatypes.TrainRequest(
        request_id="req-1",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[1]], dtype=np.int32),
            prompt_mask=np.ones((1, 1), dtype=np.float32),
            completion_ids=np.array([[2]], dtype=np.int32),
            completion_mask=np.array([[1]], dtype=np.float32),
            advantages=np.array([1.0], dtype=np.float32),
        ),
    )
    self.worker.fwd_bwd(request=train_req)
    self.assertEqual(self.events, ["enter_ctx", "exit_ctx"])

    # Verify update, checkpoints, weight sync, and stop operations.
    for op in [
        self.worker.update,
        lambda: self.worker.save_checkpoint(metadata={"step": 1}),
        lambda: self.worker.restore_checkpoint(step=1),
        self.worker.prepare_weight_sync,
        self.worker.release_weight_sync,
        self.worker.stop,
    ]:
      self.events.clear()
      op()
      self.assertEqual(self.events, ["enter_ctx", "exit_ctx"])

  def test_execution_context_exits_cleanly_on_exception(self):
    self.worker.initialize()
    self.events.clear()

    def _failing_save(*args, **kwargs):
      del args, kwargs
      self.events.append("save_failed")
      raise RuntimeError("Disk full")

    self.fake_trainer.save_checkpoint = _failing_save
    with self.assertRaisesRegex(RuntimeError, "Disk full"):
      self.worker.save_checkpoint(metadata={"step": 1})

    self.assertEqual(self.events, ["enter_ctx", "save_failed", "exit_ctx"])

  def test_repeated_stop_closes_trainer_once(self):
    self.worker.initialize()
    close_calls = []
    self.fake_trainer.close = lambda: close_calls.append(1)

    first = self.worker.stop()
    second = self.worker.stop()

    self.assertEqual(self.worker.state, datatypes.WorkerState.STOPPED)
    self.assertNotIn("already_stopped", first.metadata)
    self.assertTrue(second.metadata["already_stopped"])
    self.assertEqual(close_calls, [1])

  def test_stop_keeps_failure_signal_without_reclosing_if_close_raises(self):
    self.worker.initialize()
    close_calls = []

    def _failing_close():
      close_calls.append(1)
      raise RuntimeError("checkpoint flush failed")

    self.fake_trainer.close = _failing_close
    observed_aborts = []
    self.worker.set_on_stop(observed_aborts.append)

    # A clean (non-abort) stop whose close fails is escalated to an abort.
    with self.assertRaisesRegex(RuntimeError, "checkpoint flush failed"):
      self.worker.stop()

    self.assertEqual(self.worker.state, datatypes.WorkerState.STOPPED)
    self.assertEqual(close_calls, [1])
    self.assertEqual(observed_aborts, [True])
    self.assertEqual(
        self.worker.heartbeat().last_error, "checkpoint flush failed"
    )

    # Local teardown (`worker_service.stop()`) does not re-close the trainer,
    # but still surfaces the earlier failure with the abort flag latched.
    with self.assertRaisesRegex(
        RuntimeError, "failed to stop cleanly earlier: checkpoint flush failed"
    ):
      self.worker.stop()
    self.assertEqual(close_calls, [1])
    self.assertEqual(observed_aborts, [True, True])


def _collect_otel_metrics(reader: sdk_metrics_export.InMemoryMetricReader):
  """Extracts {metric_name: [(value_or_sum, count_or_none, dict(attributes)), ...]}."""
  data = reader.get_metrics_data()
  collected = collections.defaultdict(list)
  for resource_metric in data.resource_metrics:
    for scope_metric in resource_metric.scope_metrics:
      for metric in scope_metric.metrics:
        for dp in metric.data.data_points:
          val = dp.sum if hasattr(dp, "sum") else dp.value
          cnt = dp.count if hasattr(dp, "count") else None
          collected[metric.name].append((val, cnt, dict(dp.attributes)))
  return collected


class TrainerWorkerOTelMetricsTest(absltest.TestCase):
  """Verifies direct OpenTelemetry SDK instruments emitted by TrainerWorker."""

  def setUp(self):
    super().setUp()
    self.reader = sdk_metrics_export.InMemoryMetricReader()
    self.meter_provider = sdk_metrics.MeterProvider(
        metric_readers=[self.reader]
    )
    self.fake_trainer = FakeTrainer()
    self.worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_otel_0",
        otel_meter_provider=self.meter_provider,
    )
    self.worker.initialize()

  def test_emits_step_duration_tokens_processed_and_weight_sync_duration(self):
    # 1. Padded RLTrainerPayload: prompt (2x3, 4 valid + 2 pad),
    # completion (2x4, 5 valid + 3 pad) => total=14, valid=9, padded=5.
    padded_rl_req = datatypes.TrainRequest(
        request_id="req-padded",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),
            prompt_mask=np.array(
                [[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32
            ),
            completion_ids=np.array(
                [[10, 11, 12, 0], [20, 21, 0, 0]], dtype=np.int32
            ),
            completion_mask=np.array(
                [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], dtype=np.float32
            ),
            advantages=np.array([1.0, -1.0], dtype=np.float32),
        ),
    )
    self.worker.fwd_bwd(request=padded_rl_req)

    # 2. Sequence-packed RLTrainerPayload: prompt (1x0), completion (1x6) with
    # segment_ids [1, 1, 2, 2, 0, 0] => total=6, valid=4, padded=2.
    # Also pass `mb_ready_time_s` to verify `data_fetch_delay` starvation.
    packed_rl_req = datatypes.TrainRequest(
        request_id="req-packed",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.zeros((1, 0), dtype=np.int32),
            prompt_mask=np.zeros((1, 0), dtype=np.float32),
            completion_ids=np.array([[1, 2, 3, 4, 0, 0]], dtype=np.int32),
            completion_mask=np.array(
                [[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32
            ),
            advantages=np.zeros((1, 6), dtype=np.float32),
            segment_ids=np.array([[1, 1, 2, 2, 0, 0]], dtype=np.int32),
            segment_positions=np.array([[0, 1, 0, 1, 0, 0]], dtype=np.int32),
        ),
        metadata={"mb_ready_time_s": 0.0},
    )
    self.worker.fwd_bwd(request=packed_rl_req)

    # 3. SFTTrainerPayload: token_ids (1x4) with 3 valid + 1 pad.
    sft_req = datatypes.TrainRequest(
        request_id="req-sft",
        payload=datatypes.SFTTrainerPayload(
            token_ids=np.array([[1, 2, 3, 0]], dtype=np.int32),
            token_mask=np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32),
        ),
    )
    self.worker.fwd_bwd(request=sft_req)

    # 4. update, eval_step, per_token_logps, save_checkpoint, and
    # prepare_weight_sync.
    self.worker.update()
    self.worker.eval_step(request=sft_req)
    self.worker.per_token_logps(
        items=datatypes.LogprobsRequest(
            prompt_tokens=np.zeros((1, 0), dtype=np.int32),
            completion_tokens=np.array([[3, 4]], dtype=np.int32),
            temperature=1.0,
            pad_id=0,
            eos_id=1,
        )
    )
    self.worker.save_checkpoint(metadata={"step": 11}, step=11)
    self.worker.prepare_weight_sync()
    self.worker.release_weight_sync()

    metrics = _collect_otel_metrics(self.reader)

    # Verify tokens.processed: valid = 9 + 4 + 3 = 16, padded = 5 + 2 + 1 = 8
    tokens_by_type = {
        attrs["token_type"]: (val, attrs)
        for val, _, attrs in metrics["tunix.trainer.tokens.processed"]
    }
    self.assertEqual(
        tokens_by_type["valid"],
        (16, {"worker_id": "trainer_otel_0", "token_type": "valid"}),
    )
    self.assertEqual(
        tokens_by_type["padded"],
        (8, {"worker_id": "trainer_otel_0", "token_type": "padded"}),
    )

    # Verify step.duration phases
    step_by_phase = {
        attrs["phase"]: (val, cnt, attrs)
        for val, cnt, attrs in metrics["tunix.trainer.step.duration"]
    }
    self.assertCountEqual(
        step_by_phase.keys(),
        [
            "fwd_bwd",
            "optimizer_step",
            "eval_step",
            "per_token_logps",
            "checkpoint_save",
        ],
    )
    self.assertEqual(step_by_phase["fwd_bwd"][1], 3)
    self.assertEqual(step_by_phase["optimizer_step"][1], 1)
    self.assertEqual(step_by_phase["eval_step"][1], 1)
    self.assertEqual(step_by_phase["per_token_logps"][1], 1)
    self.assertEqual(step_by_phase["checkpoint_save"][1], 1)
    for phase_val, _, phase_attrs in step_by_phase.values():
      self.assertGreaterEqual(phase_val, 0.0)
      self.assertEqual(phase_attrs["worker_id"], "trainer_otel_0")

    # Verify weight_sync.duration (both prepare "send" and "release")
    ws_by_dir = {
        attrs["direction"]: (val, cnt, attrs)
        for val, cnt, attrs in metrics["tunix.trainer.weight_sync.duration"]
    }
    self.assertCountEqual(ws_by_dir.keys(), ["send", "release"])
    self.assertEqual(ws_by_dir["send"][1], 1)
    self.assertEqual(ws_by_dir["release"][1], 1)
    for ws_val, _, ws_attrs in ws_by_dir.values():
      self.assertGreaterEqual(ws_val, 0.0)
      self.assertEqual(ws_attrs["worker_id"], "trainer_otel_0")

    # Verify starvation.duration (`trainer_idle` on microbatches 2 & 3,
    # `data_fetch_delay` on microbatch 2 with `mb_ready_time_s`).
    starv_by_kind = {
        attrs["kind"]: (val, cnt, attrs)
        for val, cnt, attrs in metrics["tunix.trainer.starvation.duration"]
    }
    self.assertCountEqual(
        starv_by_kind.keys(), ["trainer_idle", "data_fetch_delay"]
    )
    self.assertEqual(starv_by_kind["trainer_idle"][1], 2)
    self.assertEqual(starv_by_kind["data_fetch_delay"][1], 1)
    for st_val, _, st_attrs in starv_by_kind.values():
      self.assertGreaterEqual(st_val, 0.0)
      self.assertEqual(st_attrs["worker_id"], "trainer_otel_0")

  def test_packed_sft_tokens_and_failed_fwd_bwd_edge_cases(self):
    # 1. Sequence-packed SFTTrainerPayload counts `segment_ids > 0` (3 valid, 1
    # pad) even when `token_mask` only marks 1 assistant token.
    packed_sft_req = datatypes.TrainRequest(
        request_id="req-packed-sft",
        payload=datatypes.SFTTrainerPayload(
            token_ids=np.array([[10, 11, 12, 0]], dtype=np.int32),
            token_mask=np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
            segment_ids=np.array([[1, 1, 2, 0]], dtype=np.int32),
        ),
    )
    # 2. Full-sequence [B, P+C] `segment_ids` on RLTrainerPayload with non-empty
    # prompt must not double-count prompt tokens (total=5, valid=4, padded=1).
    full_seg_rl_req = datatypes.TrainRequest(
        request_id="req-full-seg-rl",
        payload=datatypes.RLTrainerPayload(
            prompt_ids=np.array([[1, 2]], dtype=np.int32),
            prompt_mask=np.array([[1.0, 1.0]], dtype=np.float32),
            completion_ids=np.array([[3, 4, 0]], dtype=np.int32),
            completion_mask=np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
            advantages=np.zeros((1, 3), dtype=np.float32),
            segment_ids=np.array([[1, 1, 1, 1, 0]], dtype=np.int32),
        ),
    )
    orig_fwd_bwd = self.fake_trainer.fwd_bwd
    # Mock wall clock on `self.worker._otel._clock` (isolated from JAX internal
    # `time.time()` calls during cold compilation):
    # - Call 1 (`packed_sft_req`): `finally` arms free_time = 10.0
    # - Call 2 (failing `fwd_bwd`): `record_starvation` reads 12.0 (idle=2.0),
    #   then raises `RuntimeError`, and `finally` sets free_time = 15.0
    # - Call 3 (`full_seg_rl_req` after resetting `_state` to `READY`, since
    #   `WorkerState.ERROR` is terminal): `record_starvation` reads 15.5
    #   (idle = 15.5 - 15.0 = 0.5, proving the failed call's `finally` updated
    #   free_time to 15.0), and `finally` sets 16.0.
    with mock.patch.object(
        self.worker._otel,
        "_clock",
        side_effect=[10.0, 12.0, 15.0, 15.5, 16.0],
    ):
      self.worker.fwd_bwd(request=packed_sft_req)
      self.fake_trainer.fwd_bwd = mock.MagicMock(
          side_effect=RuntimeError("nan loss")
      )
      with self.assertRaisesRegex(RuntimeError, "nan loss"):
        self.worker.fwd_bwd(request=packed_sft_req)
      self.fake_trainer.fwd_bwd = orig_fwd_bwd
      self.worker._state = datatypes.WorkerState.READY
      self.worker.fwd_bwd(request=full_seg_rl_req)

    # A failing `release_weight_sync` must keep `WorkerState.SYNCING` (fail
    # closed, matching HEAD) and record no `direction="release"` sample.
    self.worker.prepare_weight_sync()
    self.assertEqual(self.worker.state, datatypes.WorkerState.SYNCING)
    self.fake_trainer.release_weight_sync = mock.MagicMock(
        side_effect=RuntimeError("release failed")
    )
    with self.assertRaisesRegex(RuntimeError, "release failed"):
      self.worker.release_weight_sync()
    self.assertEqual(self.worker.state, datatypes.WorkerState.SYNCING)

    metrics = _collect_otel_metrics(self.reader)
    tokens_by_type = {
        attrs["token_type"]: val
        for val, _, attrs in metrics["tunix.trainer.tokens.processed"]
    }
    # valid = 3 (SFT) + 4 (RL) = 7; padded = 1 (SFT) + 1 (RL) = 2.
    # The failed `fwd_bwd` recorded 0 tokens and 0 step durations.
    self.assertEqual(tokens_by_type, {"valid": 7, "padded": 2})
    step_by_phase = {
        attrs["phase"]: cnt
        for _, cnt, attrs in metrics["tunix.trainer.step.duration"]
    }
    self.assertEqual(step_by_phase, {"fwd_bwd": 2})
    ws_by_dir = {
        attrs["direction"]: cnt
        for _, cnt, attrs in metrics["tunix.trainer.weight_sync.duration"]
    }
    self.assertEqual(ws_by_dir, {"send": 1})
    starv_by_kind = {
        attrs["kind"]: (val, cnt)
        for val, cnt, attrs in metrics["tunix.trainer.starvation.duration"]
    }
    # Total idle = 2.0 (before failed call) + 0.5 (after failed call at 15.0)
    self.assertAlmostEqual(starv_by_kind["trainer_idle"][0], 2.5, places=5)
    self.assertEqual(starv_by_kind["trainer_idle"][1], 2)

  def test_starvation_values_and_non_train_rpc_busy_anchor(self):
    sft_req = datatypes.TrainRequest(
        request_id="req-sft",
        payload=datatypes.SFTTrainerPayload(
            token_ids=np.array([[1, 2]], dtype=np.int32),
            token_mask=np.array([[1.0, 1.0]], dtype=np.float32),
        ),
    )
    # Startup `compile` and initial weight sync before the first `fwd_bwd` must
    # NOT arm `last_free_time_s`, so MB1 emits zero cold-start starvation.
    self.worker.compile({"dummy": 1})
    self.worker.prepare_weight_sync()
    self.worker.release_weight_sync()
    self.assertIsNone(self.worker._otel.last_free_time_s)

    # Wall-clock sequence on `self.worker._otel._clock`:
    # 1. MB1 `fwd_bwd`: cold start (no starvation), `finally` arms -> 10.0
    # 2. MB2 (`mb_ready=8.0 < last_free=10.0`): start=10.5 -> idle=0.5,
    #    fetch_delay = 10.5 - max(10.0, 8.0) = 0.5; `finally` -> 11.0
    # 3. MB3 (`mb_ready=13.0 > last_free=11.0`): start=13.5 -> idle=2.5,
    #    fetch_delay = 13.5 - max(11.0, 13.0) = 0.5; `finally` -> 14.0
    # 4. MB4 (`mb_ready=20.0 > start=16.0`, future clock skew): start=16.0 ->
    #    idle=2.0, fetch_delay = max(0.0, 16.0 - 20.0) = 0.0; `finally` -> 17.0
    # 5. `per_token_logps`: `finally` updates armed free_time -> 25.0
    # 6. MB5 (after `per_token_logps`, `mb_ready="invalid"`): start=25.25 ->
    #    idle = 25.25 - 25.0 = 0.25 (excludes [17.0, 25.0] scoring window);
    #    `finally` -> 26.0
    # 7. `run_eval` over 2 batches: `finally` -> 30.0
    with mock.patch.object(
        self.worker._otel,
        "_clock",
        side_effect=[
            10.0,
            10.5,
            11.0,
            13.5,
            14.0,
            16.0,
            17.0,
            25.0,
            25.25,
            26.0,
            30.0,
        ],
    ):
      self.worker.fwd_bwd(request=sft_req)
      self.worker.fwd_bwd(
          request=dataclasses.replace(
              sft_req, metadata={"mb_ready_time_s": 8.0}
          )
      )
      self.worker.fwd_bwd(
          request=dataclasses.replace(
              sft_req, metadata={"mb_ready_time_s": 13.0}
          )
      )
      self.worker.fwd_bwd(
          request=dataclasses.replace(
              sft_req, metadata={"mb_ready_time_s": 20.0}
          )
      )
      self.worker.per_token_logps(
          items=datatypes.LogprobsRequest(
              prompt_tokens=np.zeros((1, 0), dtype=np.int32),
              completion_tokens=np.array([[3, 4]], dtype=np.int32),
              temperature=1.0,
              pad_id=0,
              eos_id=1,
          )
      )
      self.worker.fwd_bwd(
          request=dataclasses.replace(
              sft_req, metadata={"mb_ready_time_s": "not-a-float"}
          )
      )
      self.worker.run_eval(eval_ds=[sft_req.payload, sft_req.payload])

    metrics = _collect_otel_metrics(self.reader)
    starv_by_kind = {
        attrs["kind"]: (val, cnt)
        for val, cnt, attrs in metrics["tunix.trainer.starvation.duration"]
    }
    # trainer_idle sum = 0.5 + 2.5 + 2.0 + 0.25 = 5.25 across 4 microbatches.
    self.assertAlmostEqual(starv_by_kind["trainer_idle"][0], 5.25, places=5)
    self.assertEqual(starv_by_kind["trainer_idle"][1], 4)
    # data_fetch_delay sum = 0.5 + 0.5 + 0.0 = 1.0 across 3 valid samples.
    self.assertAlmostEqual(starv_by_kind["data_fetch_delay"][0], 1.0, places=5)
    self.assertEqual(starv_by_kind["data_fetch_delay"][1], 3)

    step_by_phase = {
        attrs["phase"]: cnt
        for _, cnt, attrs in metrics["tunix.trainer.step.duration"]
    }
    self.assertEqual(step_by_phase["eval_step"], 2)


if __name__ == "__main__":
  absltest.main()
