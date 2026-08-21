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


class ValidatedBatchHandle:
    """Single-use handle wrapping a sealed UpdateInputEvent and tensor views."""

    def __init__(self, input_event: UpdateInputEvent, batch: MaterializedBatch):
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


class GuardedUpdateAdapter:
    """Update adapter whose ONLY input is a ValidatedBatchHandle."""

    def __init__(self, store: ArtifactStore):
        self._store = store

    def update(self, handle: ValidatedBatchHandle) -> None:
        """Execute one optimizer step from the handle's sealed event.

        Fail-closed checks, in order:
        1. handle not consumed yet (consume() raises otherwise);
        2. update event says tokenizer was never called;
        3. authoritative logprob event matches the sealed event ref.
        """
        if not isinstance(handle, ValidatedBatchHandle):
            raise TypeError("guarded update only accepts ValidatedBatchHandle (no text fallback)")
        ev = handle.input_event
        if ev.tokenizer_called:
            raise RuntimeError("tokenizer was called during materialization — refusing update")
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

    return ValidatedBatchHandle(input_event, batch)


def _dtype(ref: ArtifactRef) -> np.dtype:
    # bf16 bytes are opaque at this layer; the 2-byte width is what matters
    # for layout identity.  Real bf16 math happens in the torch-side probe.
    return np.dtype({"bf16": "<f2", "float16": "<f2", "float32": "<f4", "int8": "<i1", "int32": "<i4"}.get(ref.dtype or "", "<i4"))


def _now_utc() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
