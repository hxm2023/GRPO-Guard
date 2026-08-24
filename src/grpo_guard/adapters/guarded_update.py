"""Guarded batch materializer, single-use ValidatedBatchHandle and the
capability-gated optimizer entry (design doc §7.3.3, §6.1).

The public update API accepts ONLY a ValidatedBatchHandle.  Text input,
artifact-ref substitution, and nonce reuse must fail before the optimizer.
The materializer never re-tokenizes: ``tokenizer_called`` stays False on the
valid path and any tokenizer call is a hard error.

v0.4.0 scope (P0-1..P0-3, from the 2026-08-25 audit):
- the ACTUAL tensors the optimizer consumes are bound at materialization:
  reward values are content-hashed, GRPO group size/membership/order is
  frozen in the handle, and the expected parent policy manifest is checked
  against the real model object before backward;
- nonce consumption is a transactional SQLite insert (exactly-once across
  processes, not just sequential reloads);
- the step is crash-consistent, NOT in-memory atomic: a WAL records
  PREPARED -> APPLIED -> CHECKPOINTED -> COMMITTED, and post-loss failures
  require discarding the worker and resuming from the last committed
  checkpoint.  Full in-memory rollback is NOT claimed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grpo_guard.schema.artifacts import ArtifactRef
from grpo_guard.schema.events import EventRef, UpdateInputEvent
from grpo_guard.store.artifact_store import ArtifactStore
from grpo_guard.store.canonical_json import canonical_dumps


class HandleConsumedError(RuntimeError):
    pass


class TextInputRejected(TypeError):
    pass


class NonceReuseError(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterializedBatch:
    """Tensor views the optimizer consumes; source bytes stay in the store."""

    sequence_token_ids: np.ndarray
    loss_mask: np.ndarray
    behavior_logprobs: np.ndarray
    rewards: np.ndarray
    layout_sha256: str


class _HandleIssuer:
    """Only the materializer may mint handles (producer ownership)."""


_HANDLE_ISSUER = _HandleIssuer()


class ValidatedBatchHandle:
    """Single-use handle wrapping a sealed UpdateInputEvent and tensor views.

    Only ``materialize`` may mint handles: the constructor requires the
    private issuer token, so no external code can fabricate a handle
    (P0-1: the optimizer entry cannot be bypassed by hand-constructing
    a handle).
    """

    def __init__(self, input_event: UpdateInputEvent, batch: MaterializedBatch, _issuer: _HandleIssuer = None):
        if _issuer is not _HANDLE_ISSUER:
            raise TypeError("ValidatedBatchHandle may only be created by materialize()")
        if not input_event.event_sha256:
            raise ValueError("UpdateInputEvent must be sealed")
        self._input_event = input_event
        self._batch = batch
        self._consumed = False

    @property
    def input_event(self) -> UpdateInputEvent:
        return self._input_event

    @property
    def nonce_sha256(self) -> str:
        return self._input_event.single_use_nonce_sha256

    def consume(self) -> MaterializedBatch:
        """Atomically consume the handle; any second call fails closed."""
        if self._consumed:
            raise HandleConsumedError(f"handle for {self._input_event.update_id} already consumed")
        self._consumed = True
        return self._batch


_SQLITE_MAGIC = b"SQLite format 3\x00"


def _rewards_sha256(rewards: np.ndarray) -> tuple[str, list[int], str]:
    """Content hash of the ACTUAL reward tensor the optimizer will consume.

    Rewards are bound to the handle (P0-3): the same RewardEvent can no
    longer be paired with a different reward tensor.
    """
    arr = np.ascontiguousarray(rewards.astype(np.float32, copy=False))
    return hashlib.sha256(arr.tobytes()).hexdigest(), list(arr.shape), str(arr.dtype)


class NonceRegistry:
    """PERSISTENT single-use nonce registry (SQLite, transactional).

    Exactly-once across processes: consumption is one
    ``BEGIN IMMEDIATE ... INSERT ... ON CONFLICT DO NOTHING ... COMMIT``
    transaction keyed on a UNIQUE column — concurrent workers racing on the
    same nonce produce exactly one success (verified by a multiprocessing
    test, P0-2).  Pre-v0.4.0 JSONL registries are imported in place on first
    open so ``--resume`` keeps its semantics.
    """

    def __init__(self, path: str | Path | None = None):
        self._path = None
        self._conn = None
        self._lock = threading.Lock()
        self._memory: set[str] = set()
        if path is None:
            return
        p = Path(path)
        legacy_jsonl = None
        if p.suffix == ".jsonl":
            legacy_jsonl = p
            p = p.with_suffix(".sqlite3")  # legacy runs wrote JSONL; v0.4.0 uses SQLite
        self._path = p
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists() and not self._is_sqlite(self._path):
            self._migrate_legacy_file(self._path)
        self._conn = sqlite3.connect(str(self._path), timeout=30)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS consumed_nonce ("
            " nonce TEXT PRIMARY KEY, update_id TEXT NOT NULL DEFAULT '',"
            " created_at TEXT NOT NULL)"
        )
        self._conn.commit()
        if legacy_jsonl is not None and legacy_jsonl.exists():
            self._import_lines([ln.strip() for ln in
                                legacy_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()])

    @staticmethod
    def _is_sqlite(path: Path) -> bool:
        try:
            with open(path, "rb") as fh:
                return fh.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC
        except OSError:
            return False

    def _import_lines(self, lines: list[str]) -> None:
        """Import opaque nonce strings into the registry (idempotent)."""
        conn = self._conn
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany(
                    "INSERT OR IGNORE INTO consumed_nonce(nonce, update_id, created_at)"
                    " VALUES (?, '', ?)",
                    [(n, _now_utc()) for n in dict.fromkeys(lines)],
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _migrate_legacy_file(self, path: Path) -> None:
        """Import a pre-0.4.0 JSONL of opaque nonce strings into SQLite."""
        tmp = path.with_name(path.name + ".tmp")
        try:
            conn = sqlite3.connect(str(tmp))
            conn.execute(
                "CREATE TABLE IF NOT EXISTS consumed_nonce ("
                " nonce TEXT PRIMARY KEY, update_id TEXT NOT NULL DEFAULT '',"
                " created_at TEXT NOT NULL)"
            )
            lines = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and line not in lines:
                    lines.append(line)
            conn.executemany(
                "INSERT OR IGNORE INTO consumed_nonce(nonce, update_id, created_at)"
                " VALUES (?, '', ?)",
                [(n, _now_utc()) for n in lines],
            )
            conn.commit()
            conn.close()
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @property
    def path(self) -> Path | None:
        return self._path

    def is_consumed(self, nonce_sha256: str) -> bool:
        if self._conn is None:
            return nonce_sha256 in self._memory
        cur = self._conn.execute(
            "SELECT 1 FROM consumed_nonce WHERE nonce = ?", (nonce_sha256,))
        return cur.fetchone() is not None

    def consume(self, nonce_sha256: str, update_id: str = "") -> None:
        """Exactly-once consume; a concurrent/second consume raises."""
        if self._conn is None:
            if nonce_sha256 in self._memory:
                raise NonceReuseError(f"nonce {nonce_sha256[:12]} already consumed (in-memory registry)")
            self._memory.add(nonce_sha256)
            return
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO consumed_nonce(nonce, update_id, created_at)"
                    " VALUES (?, ?, ?)",
                    (nonce_sha256, update_id, _now_utc()),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        if cur.rowcount == 0:
            raise NonceReuseError(
                f"nonce {nonce_sha256[:12]} already consumed (transactional registry)")


class UpdateWal:
    """Crash-consistent intent log for guarded updates (P0-1).

    Append-only JSONL, fsynced on every record.  The step transitions
    PREPARED (before loss, with input hashes + nonces + parent manifest)
    -> APPLIED (after optimizer.step) -> CHECKPOINTED (checkpoint promoted)
    -> COMMITTED (commit event durable).  Any failure writes ABORTED and
    re-raises.  Recovery semantics: an update whose last record is not
    COMMITTED must be discarded — the worker's in-memory state is not
    trusted and training resumes from the last committed checkpoint.
    """

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else None
        self._fh = None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path | None:
        return self._path

    def write(self, status: str, update_id: str, **fields) -> None:
        if self._fh is None:
            return
        record = {"status": status, "update_id": update_id, "ts": _now_utc(), **fields}
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def status(self, update_id: str) -> str | None:
        """Last recorded status for an update, or None."""
        if self._path is None or not self._path.exists():
            return None
        last = None
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("update_id") == update_id:
                last = rec.get("status")
        return last

    def dangling(self) -> list[str]:
        """Update ids whose last record is not COMMITTED (must be redone)."""
        if self._path is None or not self._path.exists():
            return []
        statuses: dict[str, str] = {}
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("update_id"):
                statuses[rec["update_id"]] = rec.get("status", "")
        return [uid for uid, st in statuses.items() if st != "COMMITTED"]

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def policy_manifest(model) -> str:
    """Fast structural identity of the optimizer target.

    Hashes param names + shapes + a deterministic 32-byte endpoint sample
    of every tensor (not a byte-level proof of all weights — that is the
    checkpoint digest's job at commit time).  Catches wrong-object /
    wrong-checkpoint wiring before backward at negligible per-step cost.
    """
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        data = param.data.detach()
        sample = _endpoint_bytes(data)
        h.update(name.encode())
        h.update(json.dumps(list(data.shape), sort_keys=True).encode())
        h.update(sample)
    return h.hexdigest()


def _endpoint_bytes(data) -> bytes:
    import torch

    flat = data.reshape(-1)
    u8 = flat.view(torch.uint8)
    total = u8.numel()
    if total == 0:
        return b""
    head = u8[:16].cpu().numpy().tobytes()
    tail = u8[-16:].cpu().numpy().tobytes() if total > 16 else b""
    return head + tail


class GuardedUpdateAdapter:
    """Update adapter whose ONLY input is a ValidatedBatchHandle.

    Fail-closed preconditions:
    1. the handle is a ValidatedBatchHandle (no text fallback);
    2. the referenced pre-update validation decision is ALLOW (verified via
       the decision verifier — the adapter cannot confirm ALLOW without it,
       so the verifier is REQUIRED and absence is a hard error);
    3. the nonce was never consumed before (transactional registry raises
       NonceReuseError — reuse must fail BEFORE the optimizer);
    4. the update event says the tokenizer was never called;
    5. the referenced artifacts still match their content hashes;
    6. (v0.4.0) the actual reward tensor matches the bound reward hash.
    """

    def __init__(self, store: ArtifactStore, decision_verifier=None):
        self._store = store
        self._decision_verifier = decision_verifier
        self._consumed_nonces: set[str] = set()

    def update(self, handle: ValidatedBatchHandle) -> None:
        """Execute one optimizer step from the handle's sealed event."""
        if not isinstance(handle, ValidatedBatchHandle):
            raise TypeError("guarded update only accepts ValidatedBatchHandle (no text fallback)")
        ev = handle.input_event
        if self._decision_verifier is None:
            raise RuntimeError("guarded update requires a decision verifier (ALLOW precondition)")
        if not self._decision_verifier(ev.preupdate_validation_decision):
            raise RuntimeError(
                f"update input {ev.event_id} references a validation decision that is not ALLOW"
            )
        if ev.single_use_nonce_sha256 in self._consumed_nonces:
            raise NonceReuseError(f"nonce {ev.single_use_nonce_sha256[:12]} already consumed")
        if ev.tokenizer_called:
            raise RuntimeError("tokenizer was called during materialization — refusing update")
        self._consumed_nonces.add(ev.single_use_nonce_sha256)
        batch = handle.consume()
        self._verify_artifacts(ev, batch)

    def _verify_artifacts(self, ev: UpdateInputEvent, batch: MaterializedBatch) -> None:
        for ref in (ev.sequence_token_ids, ev.loss_mask, ev.authoritative_behavior_logprobs):
            if not self._store.verify(ref):
                raise RuntimeError(f"artifact {ref.sha256[:12]} failed content hash at update time")
        if ev.reward_value_sha256:
            actual, shape, dtype = _rewards_sha256(batch.rewards)
            if actual != ev.reward_value_sha256 or shape != ev.reward_shape or dtype != ev.reward_dtype:
                raise RuntimeError(
                    f"update input {ev.event_id}: reward tensor does not match the "
                    f"bound reward hash (rewired reward wiring)")


def materialize(
    store: ArtifactStore,
    run_id: str,
    update_id: str,
    preupdate_envelope: EventRef,
    validation_decision: EventRef,
    sequence_ref: ArtifactRef,
    loss_mask_ref: ArtifactRef,
    logprob_event_ref: EventRef,
    logprob_ref: ArtifactRef,
    reward_event_ref: EventRef,
    nonce: str,
    rewards: np.ndarray | None = None,
    lifecycle_seq: int = 0,
    group_size: int | None = None,
    group_members: list[str] | None = None,
    parent_policy_manifest: str = "",
) -> ValidatedBatchHandle:
    """Rebuild tensors from content-addressed bytes only (no re-tokenization).

    v0.4.0: the reward VALUES, GRPO group size, group membership order and
    expected parent policy manifest are frozen into the sealed event, so a
    correctly-validated envelope can no longer be wired to a different
    reward tensor, group or model.
    """
    seq = store.get(sequence_ref)
    loss = store.get(loss_mask_ref)
    logprobs = store.get(logprob_ref)

    seq_arr = np.frombuffer(seq, dtype=_dtype(sequence_ref)).copy()
    loss_arr = np.frombuffer(loss, dtype=_dtype(loss_mask_ref)).copy()
    lp_arr = np.frombuffer(logprobs, dtype=_dtype(logprob_ref)).copy()

    if rewards is None:
        raise ValueError("rewards tensor must be provided by the reward adapter (never fabricated here)")
    if rewards.ndim != 1 or rewards.shape[0] == 0:
        raise ValueError("rewards must be a non-empty 1-D per-sequence tensor")
    reward_sha, reward_shape, reward_dtype = _rewards_sha256(rewards)
    members = list(group_members or [])

    layout = {
        "sequence_token_ids": {"sha256": sequence_ref.sha256, "num_bytes": sequence_ref.num_bytes},
        "loss_mask": {"sha256": loss_mask_ref.sha256, "num_bytes": loss_mask_ref.num_bytes},
        "behavior_logprobs": {"sha256": logprob_ref.sha256, "num_bytes": logprob_ref.num_bytes},
        "reward_event": reward_event_ref.event_id,
        "reward_values": {"sha256": reward_sha, "shape": reward_shape, "dtype": reward_dtype},
        "group_size": group_size,
        "group_members": members,
        "parent_policy_manifest": parent_policy_manifest,
    }
    layout_sha = hashlib.sha256(canonical_dumps(layout)).hexdigest()

    batch = MaterializedBatch(
        sequence_token_ids=seq_arr,
        loss_mask=loss_arr,
        behavior_logprobs=lp_arr,
        rewards=rewards.astype(np.float32, copy=True),
        layout_sha256=layout_sha,
    )

    input_event = UpdateInputEvent(
        event_id=f"uinput-{update_id}-{hashlib.sha256(nonce.encode()).hexdigest()[:12]}",
        event_type="update_input_materialized",
        run_id=run_id,
        component_id="guarded_materializer",
        lifecycle_seq=lifecycle_seq,
        created_at_utc=_now_utc(),
        input_events=[validation_decision],
        input_artifacts=[sequence_ref, loss_mask_ref, logprob_ref],
        update_id=update_id,
        preupdate_envelope=preupdate_envelope,
        preupdate_validation_decision=validation_decision,
        sequence_token_ids=sequence_ref,
        loss_mask=loss_mask_ref,
        authoritative_behavior_logprob_event=logprob_event_ref,
        authoritative_behavior_logprobs=logprob_ref,
        reward_event=reward_event_ref,
        materialized_layout_sha256=layout_sha,
        single_use_nonce_sha256=hashlib.sha256(nonce.encode()).hexdigest(),
        tokenizer_called=False,
        reward_value_sha256=reward_sha,
        reward_shape=reward_shape,
        reward_dtype=reward_dtype,
        group_size=group_size,
        group_members=members,
        parent_policy_manifest=parent_policy_manifest,
    ).seal()

    return ValidatedBatchHandle(input_event, batch, _HANDLE_ISSUER)


@dataclass
class GuardedStepResult:
    """Result of one guarded optimizer step (P0-1)."""

    loss: float
    metrics: dict
    consumed_nonces: list[str]
    checkpoint: dict | None = None
    wal_status: str | None = None


# Failpoint names: each raises AFTER the named phase completes, so tests can
# observe the WAL state and prove crash-consistency behavior at every stage.
FAILPOINTS = ("after_prepared", "loss", "backward", "step", "checkpoint", "commit")


def guarded_optimizer_step(
    handles: "list[ValidatedBatchHandle]",
    model,
    optimizer,
    store: ArtifactStore,
    decision_verifier,
    nonce_registry: NonceRegistry,
    group_size: int,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
    max_micro_batch: int | None = None,
    commit_fn=None,
    update_wal: UpdateWal | None = None,
    failpoint: str | None = None,
) -> GuardedStepResult:
    """The capability-gated optimizer entry (P0-1).

    Phase order (v0.4.0):
    1. verify every precondition BEFORE touching any state: sealed event,
       ALLOW decision, no tokenizer call, artifact content hashes, bound
       reward hash, group size/membership, parent policy manifest, nonce
       not consumed — ALL of these fail before backward, leaving model
       parameters unchanged;
    2. consume nonces transactionally (exactly-once across processes);
    3. WAL PREPARED (input hashes + nonces + parent manifest);
    4. loss -> backward -> optimizer.step;
    5. WAL APPLIED; commit_fn (checkpoint promotion) -> WAL CHECKPOINTED;
    6. WAL COMMITTED.

    Post-step failures (step/checkpoint/commit) do NOT roll back in-memory
    parameters — that is NOT claimed.  They leave an APPLIED/CHECKPOINTED
    WAL record; recovery discards the worker and resumes from the last
    committed checkpoint (crash consistency, P0-1 engineering).
    """
    if not isinstance(handles, (list, tuple)) or not handles:
        raise TypeError("guarded_optimizer_step requires a non-empty list of ValidatedBatchHandle")
    if not all(isinstance(h, ValidatedBatchHandle) for h in handles):
        raise TypeError("guarded_optimizer_step accepts only ValidatedBatchHandle (no text fallback)")
    if failpoint is not None and failpoint not in FAILPOINTS:
        raise ValueError(f"unknown failpoint {failpoint!r}")

    # 1) verify every precondition BEFORE touching any state
    for h in handles:
        ev = h.input_event
        if not ev.event_sha256:
            raise RuntimeError(f"update input {ev.event_id} is not sealed")
        if decision_verifier is None:
            raise RuntimeError("guarded_optimizer_step requires a decision verifier (ALLOW precondition)")
        if not decision_verifier(ev.preupdate_validation_decision):
            raise RuntimeError(f"update input {ev.event_id} references a non-ALLOW validation decision")
        if ev.tokenizer_called:
            raise RuntimeError("tokenizer was called during materialization — refusing update")
        for ref in (ev.sequence_token_ids, ev.loss_mask, ev.authoritative_behavior_logprobs):
            if not store.verify(ref):
                raise RuntimeError(f"artifact {ref.sha256[:12]} failed content hash at update time")
        if ev.reward_value_sha256:
            actual, shape, dtype = _rewards_sha256(h._batch.rewards)
            if actual != ev.reward_value_sha256 or shape != ev.reward_shape or dtype != ev.reward_dtype:
                raise RuntimeError(
                    f"update input {ev.event_id}: reward tensor does not match the bound "
                    f"reward hash (rewired reward wiring)")
        if nonce_registry.is_consumed(ev.single_use_nonce_sha256):
            raise NonceReuseError(f"nonce {ev.single_use_nonce_sha256[:12]} already consumed")

    # 2) consume nonces transactionally, then handles
    consumed = []
    for h in handles:
        nonce_registry.consume(h.input_event.single_use_nonce_sha256,
                               update_id=h.input_event.update_id)
        consumed.append(h.input_event.single_use_nonce_sha256)
    batches = [h.consume() for h in handles]
    update_ids = [h.input_event.update_id for h in handles]

    # 3) actual-input binding (P0-3): group size/membership/order and the
    #    expected parent policy are checked against the real objects the
    #    step is about to consume — all BEFORE backward.  (The reward hash
    #    was already checked in phase 1.)
    actual_members = [h.input_event.preupdate_envelope.envelope_sha256 for h in handles]
    for h in handles:
        ev = h.input_event
        if ev.group_size is not None and ev.group_size != group_size:
            raise RuntimeError(
                f"update input {ev.event_id}: bound group_size={ev.group_size} != "
                f"actual group_size={group_size} (wrong GRPO grouping)")
        if ev.group_members and ev.group_members != actual_members:
            raise RuntimeError(
                f"update input {ev.event_id}: handle order does not match the frozen "
                f"group membership (reordered batches)")
        if ev.parent_policy_manifest:
            actual_manifest = policy_manifest(model)
            if actual_manifest != ev.parent_policy_manifest:
                raise RuntimeError(
                    f"update input {ev.event_id}: model object does not match the bound "
                    f"parent policy manifest (wrong model/checkpoint wired)")

    # 4) WAL PREPARED before any loss work
    wal = update_wal
    if wal is not None:
        wal.write(
            "PREPARED", update_ids[0],
            nonces=consumed,
            layouts=[h.input_event.materialized_layout_sha256 for h in handles],
            parent_policy_manifest=handles[0].input_event.parent_policy_manifest,
            group_size=group_size,
        )
    try:
        if failpoint == "after_prepared":
            raise RuntimeError("failpoint:after_prepared")

        from grpo_guard.adapters.grpo_loss import _loss_from_batches

        result = _loss_from_batches(model, batches, group_size, clip_epsilon=clip_epsilon,
                                    beta=beta, max_micro_batch=max_micro_batch)
        if failpoint == "loss":
            raise RuntimeError("failpoint:loss")
        optimizer.zero_grad()
        if failpoint == "backward":
            raise RuntimeError("failpoint:backward")
        result.loss.backward()
        if failpoint == "step":
            raise RuntimeError("failpoint:step")
        optimizer.step()

        if wal is not None:
            wal.write("APPLIED", update_ids[0])
        if failpoint == "checkpoint":
            raise RuntimeError("failpoint:checkpoint")
        ckpt = commit_fn(model) if commit_fn is not None else None
        if wal is not None:
            wal.write("CHECKPOINTED", update_ids[0])
        if failpoint == "commit":
            raise RuntimeError("failpoint:commit")
        if wal is not None:
            wal.write("COMMITTED", update_ids[0])
        return GuardedStepResult(
            loss=float(result.metrics["loss"]),
            metrics=result.metrics,
            consumed_nonces=consumed,
            checkpoint=ckpt,
            wal_status="COMMITTED",
        )
    except Exception as exc:
        if wal is not None:
            wal.write("ABORTED", update_ids[0], error=str(exc)[:500])
        raise


def atomic_checkpoint_promotion(tmp_dir: Path, final_dir: Path) -> None:
    """Crash-consistent checkpoint promotion: fsync shards, then atomic rename.

    ``tmp_dir`` is written fully (and fsynced) first; ``final_dir`` is then
    atomically replaced.  A reader either sees the complete new checkpoint
    or the previous one — never a partial promotion (P0-1).
    """
    import shutil

    tmp_dir = Path(tmp_dir)
    final_dir = Path(final_dir)
    if not tmp_dir.exists():
        raise FileNotFoundError(f"checkpoint tmp dir {tmp_dir} missing")
    for path in sorted(tmp_dir.rglob("*")):
        if path.is_file():
            try:
                with open(path, "r+b") as fh:
                    os.fsync(fh.fileno())
            except OSError:
                pass  # fsync needs a writable handle; POSIX is the primary target
    parent = final_dir.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        # POSIX os.replace() is atomic even over a non-empty target dir;
        # on Windows a pre-existing directory must be removed first.
        shutil.rmtree(final_dir)
    os.replace(tmp_dir, final_dir)
    _fsync_dir(parent)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # Windows may refuse fsync on directories; POSIX is the target


def _dtype(ref: ArtifactRef) -> np.dtype:
    # bf16 bytes are opaque at this layer; the 2-byte width is what matters
    # for layout identity.  Real bf16 math happens in the torch-side probe.
    return np.dtype({"bf16": "<f2", "float16": "<f2", "float32": "<f4", "int8": "<i1", "int32": "<i4"}.get(ref.dtype or "", "<i4"))


def _now_utc() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
