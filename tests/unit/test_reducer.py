"""Deterministic reducer contract (design doc §7.3.4)."""

import pytest

from grpo_guard import testing
from grpo_guard.store.reducer import ConflictingCommit, reduce_sync, reduce_update


def _sync_events(t):
    return t.sync_events


def test_sync_happy_path_reducer():
    t = testing.build_trajectory()
    state = reduce_sync(t.sync_events)
    assert state.state == "CANARY_PASSED"
    assert state.terminal is True
    assert state.load_epoch == 1


def test_sync_late_callback_does_not_advance():
    t = testing.build_trajectory()
    # a late canary_passed event from the SAME attempt after terminal success
    canary = t.sync_events[-1]
    late = canary.model_copy(update={"event_id": "late-copy", "status_detail": "late"})
    late.event_sha256 = ""
    late = late.seal()
    state = reduce_sync([*t.sync_events, late])
    assert state.state == "CANARY_PASSED"
    assert state.terminal is True


def test_sync_conflicting_terminal():
    t = testing.build_trajectory()
    failed = t.sync_events[-1].model_copy(
        update={"event_id": "sync-fail", "event_type": "sync_failed"}
    )
    failed.event_sha256 = ""
    failed = failed.seal()
    with pytest.raises(ConflictingCommit):
        reduce_sync([*t.sync_events, failed])


def test_sync_new_attempt_supersedes():
    t = testing.build_trajectory()
    base = t.sync_events[-1]
    attempt2 = base.model_copy(update={
        "event_id": "attempt2-req", "event_type": "sync_requested", "attempt": 2,
        "supersedes_attempt": 1, "lifecycle_seq": base.lifecycle_seq + 1,
    })
    attempt2.event_sha256 = ""
    attempt2 = attempt2.seal()
    state = reduce_sync([*t.sync_events, attempt2])
    assert state.attempt == 2


def test_sync_late_callback_of_superseded_attempt_ignored():
    # old attempt's terminal arrives AFTER the new attempt's request
    t = testing.build_trajectory()
    base = t.sync_events[-1]
    attempt2 = base.model_copy(update={
        "event_id": "attempt2-req", "event_type": "sync_requested", "attempt": 2,
        "supersedes_attempt": 1, "lifecycle_seq": base.lifecycle_seq + 1,
    })
    attempt2.event_sha256 = ""
    attempt2 = attempt2.seal()
    late_canary = base.model_copy(update={
        "event_id": "late-canary", "event_type": "canary_passed", "attempt": 1,
        "lifecycle_seq": base.lifecycle_seq + 2,
    })
    late_canary.event_sha256 = ""
    late_canary = late_canary.seal()
    state = reduce_sync([*t.sync_events, attempt2, late_canary])
    assert state.attempt == 2
    assert state.state == "REQUESTED"  # late v0 callback did not advance v1


def test_update_reducer_committed():
    t = testing.build_trajectory()
    from grpo_guard.schema.events import UpdateEvent

    started = UpdateEvent(
        event_id="upd-start", event_type="update_started", run_id=t.run_id,
        component_id="trl_control", lifecycle_seq=50, created_at_utc=testing.now_utc(),
        update_id="update-1", transaction_id="txn-1", lease_epoch=1,
        idempotency_key="run-x:update-1", parent_policy_version=0,
    ).seal()
    committed = UpdateEvent(
        event_id="upd-commit", event_type="update_committed", run_id=t.run_id,
        component_id="trl_control", lifecycle_seq=51, created_at_utc=testing.now_utc(),
        update_id="update-1", transaction_id="txn-1", lease_epoch=1,
        idempotency_key="run-x:update-1", parent_policy_version=0, output_policy_version=1,
    ).seal()
    state = reduce_update([started, committed])
    assert state.state == "COMMITTED"
    assert state.output_policy_version == 1


def test_update_prepared_without_commit_quarantined_implicitly():
    t = testing.build_trajectory()
    from grpo_guard.schema.events import UpdateEvent

    prepared = UpdateEvent(
        event_id="upd-prep", event_type="update_prepared", run_id=t.run_id,
        component_id="trl_control", lifecycle_seq=50, created_at_utc=testing.now_utc(),
        update_id="update-1", transaction_id="txn-1", lease_epoch=1,
        idempotency_key="run-x:update-1", parent_policy_version=0,
    ).seal()
    state = reduce_update([prepared])
    assert state.state == "PREPARED"
    assert state.terminal is False  # orphan candidate, not authoritative
