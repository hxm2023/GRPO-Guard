"""Deterministic state reducers over append-only events (design doc §7.3.4).

State is derived from immutable events only — the last log line can never
overwrite prior facts.  Conflicting terminal states invalidate the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from grpo_guard.schema.events import (
    SYNC_TERMINAL_FAILURE,
    SYNC_TERMINAL_SUCCESS,
    UPDATE_TERMINAL_FAILURE,
    UPDATE_TERMINAL_SUCCESS,
    SyncEvent,
    UpdateEvent,
)


class ConflictingCommit(RuntimeError):
    """Two different committed payloads for the same ID, lease overlap, or
    conflicting terminal states → whole run is INVALID_CONFLICTING_COMMIT."""


@dataclass
class SyncState:
    sync_id: str
    state: str = "NEW"  # NEW/REQUESTED/STARTED/LOADED/CANARY_PASSED/UNKNOWN/RECONCILED_CANARY_PASSED/RETRYABLE_OLD/QUARANTINED/FAILED/SUPERSEDED
    attempt: int = 0
    load_epoch: int | None = None
    observed_policy_version: int | None = None
    terminal: bool = False
    events: list[str] = field(default_factory=list)


@dataclass
class UpdateState:
    update_id: str
    state: str = "NEW"  # NEW/STARTED/PREPARED/COMMITTED/UNKNOWN/RESTORED_PARENT/ABORTED/SUPERSEDED
    attempt: int = 0
    output_policy_version: int | None = None
    terminal: bool = False
    events: list[str] = field(default_factory=list)


def reduce_sync(events: Iterator[SyncEvent]) -> SyncState:
    """Fold sync events in lifecycle_seq order; later attempts supersede earlier."""
    state = SyncState(sync_id="")
    terminal_reached: SyncEvent | None = None
    latest_attempt = 0

    for ev in sorted(events, key=lambda e: (e.lifecycle_seq, e.event_id)):
        if not state.sync_id:
            state.sync_id = ev.sync_id
        if ev.sync_id != state.sync_id:
            raise ValueError(f"mixed sync_id in reducer: {state.sync_id} vs {ev.sync_id}")
        if ev.event_type == "sync_attempt_superseded":
            state.state = "SUPERSEDED"
            continue
        if ev.attempt < latest_attempt:
            # late callback from a superseded attempt can never advance state
            continue
        latest_attempt = ev.attempt
        if terminal_reached is not None and ev.attempt == terminal_reached.attempt:
            if ev.event_type in SYNC_TERMINAL_SUCCESS or ev.event_type in SYNC_TERMINAL_FAILURE:
                # A terminal already observed for this attempt cannot be contradicted.
                if not (ev.event_type == terminal_reached.event_type):
                    raise ConflictingCommit(f"sync {ev.sync_id} attempt {ev.attempt}: "
                                            f"{terminal_reached.event_type} then {ev.event_type}")
            continue
        state.attempt = ev.attempt
        state.events.append(ev.event_id)
        if ev.event_type in SYNC_TERMINAL_SUCCESS | SYNC_TERMINAL_FAILURE:
            terminal_reached = ev
            state.terminal = True
        if ev.observed_runtime_load_epoch is not None:
            state.load_epoch = ev.observed_runtime_load_epoch
        if ev.observed_policy_version is not None:
            state.observed_policy_version = ev.observed_policy_version
        if ev.event_type == "canary_passed":
            state.state = "CANARY_PASSED"
        elif ev.event_type == "sync_reconciled_canary_passed":
            state.state = "RECONCILED_CANARY_PASSED"
        elif ev.event_type == "sync_retryable_old":
            state.state = "RETRYABLE_OLD"
        elif ev.event_type == "sync_quarantined":
            state.state = "QUARANTINED"
        elif ev.event_type == "sync_failed":
            state.state = "FAILED"
        elif ev.event_type == "sync_unknown":
            state.state = "UNKNOWN"
        elif ev.event_type == "runtime_loaded":
            state.state = "LOADED"
        elif ev.event_type == "sync_started":
            state.state = "STARTED"
        elif ev.event_type == "sync_requested":
            state.state = "REQUESTED"
    return state


def reduce_update(events: Iterator[UpdateEvent]) -> UpdateState:
    """Fold update events; the authoritative committed payload wins conflicts."""
    state = UpdateState(update_id="")
    terminal: UpdateEvent | None = None
    latest_attempt = 0

    for ev in sorted(events, key=lambda e: (e.lifecycle_seq, e.event_id)):
        if not state.update_id:
            state.update_id = ev.update_id
        if ev.update_id != state.update_id:
            raise ValueError(f"mixed update_id in reducer: {state.update_id} vs {ev.update_id}")
        if ev.event_type == "update_attempt_superseded":
            state.state = "SUPERSEDED"
            continue
        if ev.attempt < latest_attempt:
            continue  # late callback from a superseded attempt
        latest_attempt = ev.attempt
        if terminal is not None and ev.attempt == terminal.attempt:
            if ev.event_type in UPDATE_TERMINAL_SUCCESS or ev.event_type in UPDATE_TERMINAL_FAILURE:
                if ev.event_type != terminal.event_type:
                    raise ConflictingCommit(f"update {ev.update_id} attempt {ev.attempt}: "
                                            f"{terminal.event_type} then {ev.event_type}")
            continue
        state.attempt = ev.attempt
        state.events.append(ev.event_id)
        if ev.event_type in UPDATE_TERMINAL_SUCCESS | UPDATE_TERMINAL_FAILURE:
            terminal = ev
            state.terminal = True
        if ev.output_policy_version is not None:
            state.output_policy_version = ev.output_policy_version
        if ev.event_type == "update_committed":
            state.state = "COMMITTED"
        elif ev.event_type == "update_restored_parent":
            state.state = "RESTORED_PARENT"
        elif ev.event_type == "update_aborted":
            state.state = "ABORTED"
        elif ev.event_type == "update_unknown":
            state.state = "UNKNOWN"
        elif ev.event_type == "update_prepared":
            state.state = "PREPARED"
        elif ev.event_type == "update_started":
            state.state = "STARTED"
    return state
