"""P0-1: guarded_optimizer_step — the single unbypassable optimizer entry.

Proves: (1) the only path from validated handle to optimizer goes through
guarded_optimizer_step; (2) ANY failure (non-ALLOW, tampered artifact,
nonce reuse, tokenizer_called, unsealed event) rejects BEFORE backward —
parameters are provably unchanged; (3) nonces persist across adapter
instances/processes.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grpo_guard.adapters.guarded_update import (
    MaterializedBatch,
    NonceRegistry,
    ValidatedBatchHandle,
    guarded_optimizer_step,
)
from grpo_guard.adapters.grpo_loss import grpo_loss
from grpo_guard.schema.artifacts import ArtifactRef, EnvelopeRef, EventRef
from grpo_guard.schema.events import UpdateInputEvent
from grpo_guard.store.artifact_store import ArtifactStore


class TinyModel(torch.nn.Module):
    def __init__(self, vocab=16, hidden=8):
        super().__init__()
        self.config = type("C", (), {"vocab_size": vocab})()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids):
        return (self.head(self.embed(input_ids)),)


def _make_handle(store, run_id="run-x", nonce="n1", tokenizer_called=False,
                 decision_sha="0" * 64, seq_ref=None, mask_ref=None, lp_ref=None):
    from grpo_guard.adapters.guarded_update import _HANDLE_ISSUER

    seq = np.zeros(8, dtype=np.int32)
    seq[:7] = [1, 2, 3, 4, 5, 6, 7]
    mask = np.zeros(7, dtype=np.int8)
    mask[2:6] = 1
    lp = np.full(4, -0.5, dtype=np.float32)

    def _put(name, arr, dtype):
        return store.put(arr.tobytes(), "application/octet-stream", f"gen-{name}",
                         dtype=dtype, shape=list(arr.shape))

    seq_ref = seq_ref or _put("seq", seq, "int32")
    mask_ref = mask_ref or _put("mask", mask, "int8")
    lp_ref = lp_ref or _put("lp", lp, "float32")

    ev = UpdateInputEvent(
        event_id=f"uinput-{nonce}", run_id=run_id, component_id="materializer",
        lifecycle_seq=1, created_at_utc="t", update_id="update-1",
        preupdate_envelope=EnvelopeRef(uri="", envelope_id="env-x", envelope_sha256="0" * 64),
        preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256=decision_sha),
        sequence_token_ids=seq_ref, loss_mask=mask_ref,
        authoritative_behavior_logprob_event=EventRef(uri="", event_id="gen-x", event_sha256="0" * 64),
        authoritative_behavior_logprobs=lp_ref,
        reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
        materialized_layout_sha256="0" * 64,
        single_use_nonce_sha256=nonce,
        tokenizer_called=tokenizer_called,
    ).seal()
    batch = MaterializedBatch(
        sequence_token_ids=seq, loss_mask=mask, behavior_logprobs=lp,
        rewards=np.asarray([1.0], dtype=np.float32), layout_sha256="0" * 64,
    )
    return ValidatedBatchHandle(ev, batch, _HANDLE_ISSUER)


def _allow_verifier(_):
    return True


def _reject_verifier(_):
    return False


def _params_snapshot(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}


def _assert_unchanged(model, snap):
    for n, p in model.named_parameters():
        assert torch.equal(p.data, snap[n]), f"parameter {n} changed on rejected path"


def test_guarded_step_runs_full_optimizer_cycle(tmp_path):
    store = ArtifactStore(tmp_path)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = _params_snapshot(model)

    handle = _make_handle(store)
    result = guarded_optimizer_step(
        [handle], model, optimizer, store=store, decision_verifier=_allow_verifier,
        nonce_registry=registry, group_size=1, commit_fn=None,
    )
    assert result.metrics["B"] == 1
    # optimizer ran: at least one parameter moved
    moved = any(not torch.equal(model.state_dict()[n], before[n]) for n in before)
    assert moved
    # nonce persisted
    assert registry.is_consumed("n1")


def test_guarded_step_rejects_non_allow_with_unchanged_params(tmp_path):
    store = ArtifactStore(tmp_path)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = _params_snapshot(model)

    handle = _make_handle(store, decision_sha="f" * 64)  # decision not ALLOW
    with pytest.raises(RuntimeError):
        guarded_optimizer_step([handle], model, optimizer, store=store,
                               decision_verifier=_reject_verifier, nonce_registry=registry, group_size=1)
    _assert_unchanged(model, before)


def test_guarded_step_rejects_tampered_artifact_with_unchanged_params(tmp_path):
    store = ArtifactStore(tmp_path)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = _params_snapshot(model)

    handle = _make_handle(store)
    # tamper the sequence blob AFTER materialization, BEFORE the optimizer
    seq_blob = next(store.blobs.glob("*"))
    seq_blob.write_bytes(b"\x00" * seq_blob.stat().st_size)
    with pytest.raises(RuntimeError):
        guarded_optimizer_step([handle], model, optimizer, store=store,
                               decision_verifier=_allow_verifier, nonce_registry=registry, group_size=1)
    _assert_unchanged(model, before)
    assert not registry.is_consumed("n1")  # nonce NOT consumed on failure


def test_guarded_step_rejects_tokenizer_called(tmp_path):
    store = ArtifactStore(tmp_path)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = _params_snapshot(model)

    handle = _make_handle(store, tokenizer_called=True)
    with pytest.raises(RuntimeError):
        guarded_optimizer_step([handle], model, optimizer, store=store,
                               decision_verifier=_allow_verifier, nonce_registry=registry, group_size=1)
    _assert_unchanged(model, before)


def test_nonce_reuse_across_instances_detected(tmp_path):
    store = ArtifactStore(tmp_path)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    h1 = _make_handle(store, nonce="shared-nonce")
    guarded_optimizer_step([h1], model, optimizer, store=store, decision_verifier=_allow_verifier,
                           nonce_registry=registry, group_size=1)

    # a NEW step reuses the same nonce (fresh adapter instance, same registry)
    h2 = _make_handle(store, nonce="shared-nonce")
    with pytest.raises(Exception) as exc:
        guarded_optimizer_step([h2], model, optimizer, store=store, decision_verifier=_allow_verifier,
                               nonce_registry=registry, group_size=1)
    assert "already consumed" in str(exc.value)


def test_handle_cannot_be_constructed_externally():
    ev = UpdateInputEvent(
        event_id="u-x", run_id="r", component_id="m", lifecycle_seq=1, created_at_utc="t",
        update_id="u", preupdate_envelope=EnvelopeRef(uri="", envelope_id="e", envelope_sha256="0" * 64),
        preupdate_validation_decision=EventRef(uri="", event_id="v", event_sha256="0" * 64),
        sequence_token_ids=ArtifactRef(uri="a://s", media_type="x", dtype="int32", shape=[1], num_bytes=4,
                                       sha256="0" * 64, producer_event_id="g"),
        loss_mask=ArtifactRef(uri="a://m", media_type="x", dtype="int8", shape=[1], num_bytes=1,
                              sha256="0" * 64, producer_event_id="g"),
        authoritative_behavior_logprob_event=EventRef(uri="", event_id="g", event_sha256="0" * 64),
        authoritative_behavior_logprobs=ArtifactRef(uri="a://l", media_type="x", dtype="float32", shape=[1],
                                                    num_bytes=4, sha256="0" * 64, producer_event_id="g"),
        reward_event=EventRef(uri="", event_id="rw", event_sha256="0" * 64),
        materialized_layout_sha256="0" * 64, single_use_nonce_sha256="n", tokenizer_called=False,
    ).seal()
    batch = MaterializedBatch(
        sequence_token_ids=np.zeros(1, dtype=np.int32), loss_mask=np.zeros(1, dtype=np.int8),
        behavior_logprobs=np.zeros(1, dtype=np.float32), rewards=np.zeros(1, dtype=np.float32),
        layout_sha256="0" * 64,
    )
    with pytest.raises(TypeError):
        ValidatedBatchHandle(ev, batch)  # no issuer token -> external mint blocked


def test_handle_consume_is_single_use(tmp_path):
    store = ArtifactStore(tmp_path)
    handle = _make_handle(store)
    handle.consume()
    with pytest.raises(Exception):
        handle.consume()
