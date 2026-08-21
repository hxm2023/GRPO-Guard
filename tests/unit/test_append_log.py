"""Append-only log + fencing lease contract (design doc §7.3.4)."""

import pytest

from grpo_guard import testing
from grpo_guard.store.append_log import AppendLog, EventAlreadyAppended, LeaseError
from grpo_guard.store.canonical_json import canonical_dumps


@pytest.fixture()
def log(tmp_path):
    return AppendLog(tmp_path / "log", run_id="run-x", lease_id="writer-1")


def _event(log, seq):
    from grpo_guard.schema.events import EventBase

    ev = EventBase(
        event_id=f"ev-{seq}",
        event_type="test",
        run_id="run-x",
        component_id="t",
        lifecycle_seq=seq,
        created_at_utc=testing.now_utc(),
    ).seal()
    return ev


def test_append_requires_lease(log):
    with pytest.raises(LeaseError):
        log.append(_event(log, 0))


def test_lease_fencing(log):
    assert log.acquire_lease() == 1
    assert log.acquire_lease() == 2  # same writer bumps epoch
    other = AppendLog(log.root, run_id="run-x", lease_id="writer-2")
    with pytest.raises(LeaseError):
        other.acquire_lease()
    # original writer still holds; epoch moved on → stale writer fails
    log.append(_event(log, 0), required_epoch=2)
    with pytest.raises(LeaseError):
        log.append(_event(log, 1), required_epoch=1)


def test_append_requires_sealed(log):
    from grpo_guard.schema.events import EventBase

    log.acquire_lease()
    ev = EventBase(event_id="unsealed", event_type="test", run_id="run-x", component_id="t",
                   lifecycle_seq=0, created_at_utc=testing.now_utc())
    with pytest.raises(ValueError):
        log.append(ev)


def test_idempotent_append(log):
    log.acquire_lease()
    ev = _event(log, 0)
    log.append(ev)
    log.append(ev)  # same payload → idempotent
    with pytest.raises(EventAlreadyAppended):
        tampered = ev.model_copy(update={"created_at_utc": "different"})
        tampered.event_sha256 = ""
        log.append(tampered.seal())


def test_append_is_canonical_json(log):
    log.acquire_lease()
    ev = _event(log, 0)
    log.append(ev)
    raw = (log.events_dir / "ev-0.json").read_bytes()
    assert raw == canonical_dumps(ev.model_dump(mode="json"))


def test_provenance_edge_conflict(log):
    log.acquire_lease()
    log.append_provenance_edge("ev-0", "abc", "producer")
    log.append_provenance_edge("ev-0", "abc", "producer")  # idempotent
    with pytest.raises(ValueError):
        log.append_provenance_edge("ev-0", "abc", "consumer")
