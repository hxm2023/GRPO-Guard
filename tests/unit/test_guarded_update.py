"""Guarded update contract (design doc §7.3.3, §6.1)."""

import numpy as np
import pytest

from grpo_guard import testing
from grpo_guard.adapters.guarded_update import (
    GuardedUpdateAdapter,
    HandleConsumedError,
    NonceReuseError,
    TextInputRejected,
    materialize,
)
from grpo_guard.schema.artifacts import EventRef


def _handle(t, rewards=None, nonce="nonce-1"):
    gen = t.events[t.envelope.generation_event.event_id]
    if rewards is None:
        rewards = np.zeros(gen.loss_mask.shape[0], dtype=np.float32)
    return materialize(
        store=t.store,
        run_id=t.run_id,
        update_id="update-1",
        preupdate_envelope=t.envelope.ref(),
        validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
        sequence_ref=gen.sequence_token_ids,
        loss_mask_ref=gen.loss_mask,
        logprob_event_ref=t.envelope.training_contract.authoritative_behavior_logprob_event,
        logprob_ref=gen.service_behavior_logprobs,
        reward_event_ref=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
        nonce=nonce,
        rewards=rewards,
        lifecycle_seq=t.next_seq(),
    )


def test_handle_single_use():
    t = testing.build_trajectory()
    handle = _handle(t)
    batch = handle.consume()
    assert batch.sequence_token_ids.shape[0] == 12
    with pytest.raises(HandleConsumedError):
        handle.consume()


def _allow_verifier(ref):
    return ref is not None and ref.event_id == "vdec-x" or getattr(ref, "event_id", "").startswith("vdec-")


def test_update_rejects_text_input():
    t = testing.build_trajectory()
    adapter = GuardedUpdateAdapter(t.store, decision_verifier=_allow_verifier)
    with pytest.raises(TypeError):
        adapter.update("prompt+completion text")  # no text fallback


def test_update_requires_decision_verifier():
    t = testing.build_trajectory()
    handle = _handle(t)
    adapter = GuardedUpdateAdapter(t.store)  # no verifier -> fail closed
    with pytest.raises(RuntimeError):
        adapter.update(handle)


def test_update_rejects_non_allow_decision():
    t = testing.build_trajectory()
    handle = _handle(t)
    adapter = GuardedUpdateAdapter(t.store, decision_verifier=lambda ref: False)
    with pytest.raises(RuntimeError):
        adapter.update(handle)


def test_update_consumes_handle_once():
    t = testing.build_trajectory()
    handle = _handle(t)
    adapter = GuardedUpdateAdapter(t.store, decision_verifier=_allow_verifier)
    adapter.update(handle)
    # the nonce registry fires before the handle-consume check: reuse must
    # fail BEFORE the optimizer (design doc §7.3.3)
    with pytest.raises(NonceReuseError):
        adapter.update(handle)


def test_nonce_reuse_rejected():
    t = testing.build_trajectory()
    adapter = GuardedUpdateAdapter(t.store, decision_verifier=_allow_verifier)
    h1 = _handle(t, nonce="nonce-dup")
    h2 = _handle(t, nonce="nonce-dup")  # same nonce, different event
    adapter.update(h1)
    with pytest.raises(NonceReuseError):
        adapter.update(h2)


def test_update_fails_closed_on_tokenizer_call():
    t = testing.build_trajectory()
    from grpo_guard.adapters.guarded_update import MaterializedBatch, ValidatedBatchHandle
    from grpo_guard.schema.events import UpdateInputEvent

    ev = UpdateInputEvent(
        event_id="uinput-tokenizer", run_id=t.run_id, component_id="materializer",
        lifecycle_seq=1, created_at_utc=testing.now_utc(),
        update_id="update-1",
        preupdate_envelope=t.envelope.ref(),
        preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
        sequence_token_ids=t.sequence_ref,
        loss_mask=_handle(t).input_event.loss_mask,
        authoritative_behavior_logprob_event=t.envelope.training_contract.authoritative_behavior_logprob_event,
        authoritative_behavior_logprobs=_handle(t).input_event.authoritative_behavior_logprobs,
        reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
        materialized_layout_sha256="0" * 64,
        single_use_nonce_sha256="0" * 64,
        tokenizer_called=True,  # fault: tokenizer was invoked during materialization
    ).seal()
    batch = MaterializedBatch(
        sequence_token_ids=np.zeros(1, dtype=np.int32), loss_mask=np.zeros(1, dtype=np.int8),
        behavior_logprobs=np.zeros(1, dtype=np.float32), rewards=np.zeros(1, dtype=np.float32),
        layout_sha256="0" * 64,
    )
    handle = ValidatedBatchHandle(ev, batch)
    adapter = GuardedUpdateAdapter(t.store, decision_verifier=_allow_verifier)
    with pytest.raises(RuntimeError):
        adapter.update(handle)


def test_materialize_requires_rewards():
    t = testing.build_trajectory()
    gen = t.events[t.envelope.generation_event.event_id]
    with pytest.raises(ValueError):
        materialize(
            store=t.store, run_id=t.run_id, update_id="update-1",
            preupdate_envelope=t.envelope.ref(),
            validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
            sequence_ref=gen.sequence_token_ids, loss_mask_ref=gen.loss_mask,
            logprob_event_ref=t.envelope.training_contract.authoritative_behavior_logprob_event,
            logprob_ref=gen.service_behavior_logprobs,
            reward_event_ref=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
            nonce="nonce-x", rewards=None, lifecycle_seq=t.next_seq(),
        )


def test_materialize_never_calls_tokenizer():
    t = testing.build_trajectory()
    handle = _handle(t)
    assert handle.input_event.tokenizer_called is False
    assert handle.input_event.verify_seal()


def test_materialize_rebuilds_same_bytes():
    t = testing.build_trajectory()
    handle = _handle(t)
    gen = t.events[t.envelope.generation_event.event_id]
    batch = handle.input_event
    assert batch.sequence_token_ids.sha256 == gen.sequence_token_ids.sha256
    assert batch.loss_mask.sha256 == gen.loss_mask.sha256
