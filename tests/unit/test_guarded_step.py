"""P0-1/2/3: guarded_optimizer_step — the capability-gated optimizer entry.

Proves: (1) the only path from validated handle to optimizer goes through
guarded_optimizer_step; (2) precondition failures (non-ALLOW, tampered
artifact, reward substitution, wrong group/model, nonce reuse,
tokenizer_called, unsealed event) reject BEFORE backward — parameters are
provably unchanged; (3) nonces are exactly-once across PROCESSES (SQLite
transactional registry); (4) post-loss failures follow crash consistency
(WAL PREPARED -> APPLIED -> CHECKPOINTED -> COMMITTED / ABORTED, worker
must be discarded) — full in-memory rollback is NOT claimed.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

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


# ---------------------------------------------------------------- P0-2 nonce
def test_nonce_registry_survives_reopen(tmp_path):
    db = tmp_path / "nonces.jsonl"
    r1 = NonceRegistry(db)
    r1.consume("abc")
    r2 = NonceRegistry(db)  # fresh instance, same file (new process analog)
    assert r2.is_consumed("abc")
    with pytest.raises(Exception) as exc:
        r2.consume("abc")
    assert "already consumed" in str(exc.value)


def test_nonce_registry_multiprocess_race_exactly_once(tmp_path):
    """32 processes race on the SAME nonce; exactly one wins (P0-2).

    Requires fork (POSIX): the CI Linux runs cover it; on Windows the
    spawn-based child would re-import torch per child (heavy/racy), so we
    skip locally.
    """
    import importlib.util
    import multiprocessing as mp

    if "fork" not in mp.get_all_start_methods():
        pytest.skip("multiprocessing race test requires fork (POSIX) — covered by Linux CI")

    worker_path = Path(__file__).parent / "_nonce_race_worker.py"
    spec = importlib.util.spec_from_file_location("nonce_race_worker", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    db = str(tmp_path / "nonces.sqlite3")
    NonceRegistry(db)  # create schema
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=worker.try_consume, args=(db, "race-nonce")) for _ in range(32)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
    wins = sum(1 for p in procs if p.exitcode == 1)
    assert wins == 1, f"expected exactly one winner, got {wins}"
    assert NonceRegistry(db).is_consumed("race-nonce")


# ---------------------------------------------------------------- P0-1 WAL
from grpo_guard.adapters.guarded_update import UpdateWal  # noqa: E402


def _wal_path(tmp_path):
    return tmp_path / "update_wal.jsonl"


def test_wal_happy_path_records_full_lifecycle(tmp_path):
    store = ArtifactStore(tmp_path)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    wal = UpdateWal(_wal_path(tmp_path))
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    handle = _make_handle(store, nonce="wal-nonce")
    guarded_optimizer_step([handle], model, optimizer, store=store,
                           decision_verifier=_allow_verifier, nonce_registry=registry,
                           group_size=1, update_wal=wal)
    assert wal.status("update-1") == "COMMITTED"
    assert wal.dangling() == []
    wal.close()


@pytest.mark.parametrize("failpoint", ["loss", "backward", "step"])
def test_wal_failpoint_before_step_leaves_params_unchanged(tmp_path, failpoint):
    store = ArtifactStore(tmp_path)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    wal = UpdateWal(_wal_path(tmp_path))
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = _params_snapshot(model)
    handle = _make_handle(store, nonce=f"fp-{failpoint}")
    with pytest.raises(RuntimeError, match=f"failpoint:{failpoint}"):
        guarded_optimizer_step([handle], model, optimizer, store=store,
                               decision_verifier=_allow_verifier, nonce_registry=registry,
                               group_size=1, update_wal=wal, failpoint=failpoint)
    _assert_unchanged(model, before)
    assert wal.status("update-1") == "ABORTED"
    wal.close()


def test_wal_failpoint_checkpoint_params_moved_worker_must_discard(tmp_path):
    """step already ran -> params moved; WAL is not COMMITTED -> the worker
    must be discarded and training resumed from the last committed
    checkpoint (crash consistency, NOT in-memory rollback)."""
    store = ArtifactStore(tmp_path)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    wal = UpdateWal(_wal_path(tmp_path))
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = _params_snapshot(model)
    handle = _make_handle(store, nonce="fp-ckpt")
    with pytest.raises(RuntimeError, match="failpoint:checkpoint"):
        guarded_optimizer_step([handle], model, optimizer, store=store,
                               decision_verifier=_allow_verifier, nonce_registry=registry,
                               group_size=1, update_wal=wal, failpoint="checkpoint")
    moved = any(not torch.equal(model.state_dict()[n], before[n]) for n in before)
    assert moved, "optimizer.step ran before the checkpoint failpoint"
    assert wal.status("update-1") == "ABORTED"
    assert wal.dangling() == ["update-1"]  # recovery must redo/abort this update
    wal.close()


def test_wal_unknown_failpoint_rejected(tmp_path):
    store = ArtifactStore(tmp_path)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    handle = _make_handle(store)
    with pytest.raises(ValueError, match="unknown failpoint"):
        guarded_optimizer_step([handle], model, optimizer, store=store,
                               decision_verifier=_allow_verifier, nonce_registry=registry,
                               group_size=1, failpoint="bogus")


# ---------------------------------------------------------------- P0-3 binding
from grpo_guard import testing  # noqa: E402
from grpo_guard.adapters.guarded_update import (  # noqa: E402
    _HANDLE_ISSUER,
    materialize,
    policy_manifest,
)


def _real_handle(t, rewards=None, nonce="nonce-p03", **kw):
    gen = t.events[t.envelope.generation_event.event_id]
    if rewards is None:
        rewards = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    return materialize(
        store=t.store, run_id=t.run_id, update_id="update-p03",
        preupdate_envelope=t.envelope.ref(),
        validation_decision=EventRef(uri="", event_id="vdec-p03", event_sha256="0" * 64),
        sequence_ref=gen.sequence_token_ids, loss_mask_ref=gen.loss_mask,
        logprob_event_ref=t.envelope.training_contract.authoritative_behavior_logprob_event,
        logprob_ref=gen.service_behavior_logprobs,
        reward_event_ref=EventRef(uri="", event_id="reward-p03", event_sha256="0" * 64),
        nonce=nonce, rewards=rewards, lifecycle_seq=1, **kw,
    )


def test_materialize_binds_reward_hash_and_nonce_hash(tmp_path):
    t = testing.build_trajectory()
    handle = _real_handle(t)
    ev = handle.input_event
    assert ev.reward_value_sha256 == hashlib.sha256(
        np.ascontiguousarray(np.asarray([1.0, 0.0, 1.0], dtype=np.float32)).tobytes()).hexdigest()
    # P0-3: single_use_nonce_sha256 stores the REAL SHA-256, not the raw nonce
    assert ev.single_use_nonce_sha256 == hashlib.sha256(b"nonce-p03").hexdigest()
    assert ev.single_use_nonce_sha256 != "nonce-p03"


def test_guarded_step_rejects_reward_substitution_before_backward(tmp_path):
    """Same sealed event, different actual reward tensor -> fail before
    backward (the 'same RewardEvent, different reward values' wiring bug)."""
    t = testing.build_trajectory()
    handle = _real_handle(t)
    ev = handle.input_event
    tampered = MaterializedBatch(
        sequence_token_ids=handle._batch.sequence_token_ids,
        loss_mask=handle._batch.loss_mask,
        behavior_logprobs=handle._batch.behavior_logprobs,
        rewards=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),  # substituted
        layout_sha256=handle._batch.layout_sha256,
    )
    forged = ValidatedBatchHandle(ev, tampered, _HANDLE_ISSUER)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = _params_snapshot(model)
    with pytest.raises(RuntimeError, match="reward tensor does not match"):
        guarded_optimizer_step([forged], model, optimizer, store=t.store,
                               decision_verifier=_allow_verifier, nonce_registry=registry,
                               group_size=1)
    _assert_unchanged(model, before)


def test_guarded_step_rejects_wrong_group_size(tmp_path):
    t = testing.build_trajectory()
    handle = _real_handle(t, group_size=4)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = _params_snapshot(model)
    with pytest.raises(RuntimeError, match="bound group_size"):
        guarded_optimizer_step([handle], model, optimizer, store=t.store,
                               decision_verifier=_allow_verifier, nonce_registry=registry,
                               group_size=2)  # caller wired a different grouping
    _assert_unchanged(model, before)


def test_guarded_step_rejects_wrong_model_object(tmp_path):
    """Handle bound to model A's manifest; step runs against model B."""
    t = testing.build_trajectory()
    torch.manual_seed(0)
    model_a = TinyModel()
    manifest_a = policy_manifest(model_a)
    handle = _real_handle(t, parent_policy_manifest=manifest_a)
    torch.manual_seed(1)
    model_b = TinyModel()  # different init -> different manifest
    assert policy_manifest(model_b) != manifest_a
    optimizer = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    before = _params_snapshot(model_b)
    with pytest.raises(RuntimeError, match="parent policy manifest"):
        guarded_optimizer_step([handle], model_b, optimizer, store=t.store,
                               decision_verifier=_allow_verifier, nonce_registry=registry,
                               group_size=1)
    _assert_unchanged(model_b, before)


def test_guarded_step_rejects_group_reorder(tmp_path):
    """Handles reordered against the frozen group membership -> fail."""
    t = testing.build_trajectory()
    h1 = _real_handle(t, nonce="n-a", group_members=["env-a", "env-b"])
    h2 = _real_handle(t, nonce="n-b", group_members=["env-a", "env-b"])
    registry = NonceRegistry(tmp_path / "nonces.jsonl")
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = _params_snapshot(model)
    with pytest.raises(RuntimeError, match="group membership"):
        guarded_optimizer_step([h2, h1], model, optimizer, store=t.store,
                               decision_verifier=_allow_verifier, nonce_registry=registry,
                               group_size=1)  # order swapped
    _assert_unchanged(model, before)


def test_atomic_checkpoint_promotion(tmp_path):
    from grpo_guard.adapters.guarded_update import atomic_checkpoint_promotion

    tmp = tmp_path / "ck.tmp"
    final = tmp_path / "ck"
    tmp.mkdir()
    (tmp / "shard-1.bin").write_bytes(b"x" * 100)
    atomic_checkpoint_promotion(tmp, final)
    assert (final / "shard-1.bin").read_bytes() == b"x" * 100
    assert not tmp.exists()
    # promoting over an existing checkpoint replaces it completely
    tmp2 = tmp_path / "ck2.tmp"
    tmp2.mkdir()
    (tmp2 / "shard-2.bin").write_bytes(b"y" * 50)
    atomic_checkpoint_promotion(tmp2, final)
    assert (final / "shard-2.bin").read_bytes() == b"y" * 50
    assert not (final / "shard-1.bin").exists()
    assert not tmp2.exists()
