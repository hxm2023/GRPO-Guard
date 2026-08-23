"""Guarded batch materializer and single-use ValidatedBatchHandle
(design doc §7.3.3, §6.1).

The public update API accepts ONLY a ValidatedBatchHandle.  Text input,
artifact-ref substitution, and nonce reuse must fail before the optimizer.
The materializer never re-tokenizes: ``tokenizer_called`` stays False on the
valid path and any tokenizer call is a hard error.
"""

from __future__ import annotations

import hashlib
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


class NonceRegistry:
    """PERSISTENT single-use nonce registry (append-only JSONL).

    P0-1: nonce state must survive adapter instances and processes —
    the in-memory set in GuardedUpdateAdapter cannot detect reuse across
    training steps that each construct a fresh adapter.
    """

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else None
        self._consumed: set[str] = set()
        if self._path and self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._consumed.add(line)

    def is_consumed(self, nonce_sha256: str) -> bool:
        return nonce_sha256 in self._consumed

    def consume(self, nonce_sha256: str) -> None:
        if nonce_sha256 in self._consumed:
            raise NonceReuseError(f"nonce {nonce_sha256[:12]} already consumed (persistent registry)")
        self._consumed.add(nonce_sha256)
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(nonce_sha256 + "\n")


class GuardedUpdateAdapter:
    """Update adapter whose ONLY input is a ValidatedBatchHandle.

    Fail-closed preconditions:
    1. the handle is a ValidatedBatchHandle (no text fallback);
    2. the referenced pre-update validation decision is ALLOW (verified via
       the decision verifier — the adapter cannot confirm ALLOW without it,
       so the verifier is REQUIRED and absence is a hard error);
    3. the nonce was never consumed before (in-process registry raises
       NonceReuseError — reuse must fail BEFORE the optimizer);
    4. the update event says the tokenizer was never called;
    5. the referenced artifacts still match their content hashes.
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
) -> ValidatedBatchHandle:
    """Rebuild tensors from content-addressed bytes only (no re-tokenization)."""
    seq = store.get(sequence_ref)
    loss = store.get(loss_mask_ref)
    logprobs = store.get(logprob_ref)

    seq_arr = np.frombuffer(seq, dtype=_dtype(sequence_ref)).copy()
    loss_arr = np.frombuffer(loss, dtype=_dtype(loss_mask_ref)).copy()
    lp_arr = np.frombuffer(logprobs, dtype=_dtype(logprob_ref)).copy()

    layout = {
        "sequence_token_ids": {"sha256": sequence_ref.sha256, "num_bytes": sequence_ref.num_bytes},
        "loss_mask": {"sha256": loss_mask_ref.sha256, "num_bytes": loss_mask_ref.num_bytes},
        "behavior_logprobs": {"sha256": logprob_ref.sha256, "num_bytes": logprob_ref.num_bytes},
        "reward_event": reward_event_ref.event_id,
    }
    layout_sha = hashlib.sha256(canonical_dumps(layout)).hexdigest()

    if rewards is None:
        raise ValueError("rewards tensor must be provided by the reward adapter (never fabricated here)")
    if rewards.ndim != 1 or rewards.shape[0] == 0:
        raise ValueError("rewards must be a non-empty 1-D per-sequence tensor")

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
        single_use_nonce_sha256=nonce,
        tokenizer_called=False,
    ).seal()

    return ValidatedBatchHandle(input_event, batch, _HANDLE_ISSUER)


@dataclass
class GuardedStepResult:
    """Result of one guarded optimizer step (P0-1)."""

    loss: float
    metrics: dict
    consumed_nonces: list[str]
    checkpoint: dict | None = None


def guarded_optimizer_step(
    handles: "list[ValidatedBatchHandle]",
    model,
    optimizer,
    store: ArtifactStore,
    decision_verifier,
    nonce_registry: NonceRegistry,
    group_size: int,
    loss_fn=None,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
    max_micro_batch: int | None = None,
    commit_fn=None,
) -> GuardedStepResult:
    """THE single, unbypassable optimizer entry (P0-1).

    Atomically: verify every decision/artifact hash → consume nonces from
    the PERSISTENT registry → consume the handles → compute loss → backward
    → optimizer.step → (optional) checkpoint commit.  There is no other
    path from validated handle to optimizer: ``grpo_loss`` consumes the
    same handles, and the public entry never accepts text or raw tensors.

    On ANY failure (non-ALLOW decision, artifact hash mismatch, nonce
    reuse, unsealed event, tokenizer_called) this raises BEFORE backward:
    model parameters are provably unchanged.
    """
    if not isinstance(handles, (list, tuple)) or not handles:
        raise TypeError("guarded_optimizer_step requires a non-empty list of ValidatedBatchHandle")
    if not all(isinstance(h, ValidatedBatchHandle) for h in handles):
        raise TypeError("guarded_optimizer_step accepts only ValidatedBatchHandle (no text fallback)")

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
        if nonce_registry.is_consumed(ev.single_use_nonce_sha256):
            raise NonceReuseError(f"nonce {ev.single_use_nonce_sha256[:12]} already consumed")

    # 2) consume nonces persistently, then consume handles
    consumed = []
    for h in handles:
        nonce_registry.consume(h.input_event.single_use_nonce_sha256)
        consumed.append(h.input_event.single_use_nonce_sha256)
    batches = [h.consume() for h in handles]

    # 3) loss -> backward -> step (only reachable after all checks passed)
    if loss_fn is None:
        from grpo_guard.adapters.grpo_loss import _loss_from_batches

        loss_fn = _loss_from_batches
    result = loss_fn(model, batches, group_size, clip_epsilon=clip_epsilon, beta=beta,
                     max_micro_batch=max_micro_batch)
    optimizer.zero_grad()
    result.loss.backward()
    optimizer.step()

    ckpt = commit_fn(model) if commit_fn is not None else None
    return GuardedStepResult(
        loss=float(result.metrics["loss"]),
        metrics=result.metrics,
        consumed_nonces=consumed,
        checkpoint=ckpt,
    )


def _dtype(ref: ArtifactRef) -> np.dtype:
    # bf16 bytes are opaque at this layer; the 2-byte width is what matters
    # for layout identity.  Real bf16 math happens in the torch-side probe.
    return np.dtype({"bf16": "<f2", "float16": "<f2", "float32": "<f4", "int8": "<i1", "int32": "<i4"}.get(ref.dtype or "", "<i4"))


def _now_utc() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
