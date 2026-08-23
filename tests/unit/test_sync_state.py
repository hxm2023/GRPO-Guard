"""P0-2: sync state machine — runtime_loaded only after real calls;
canary mismatch never recorded as canary_passed; no-op sync detection."""

from __future__ import annotations

import json

import pytest

from grpo_guard.adapters.trl_control import TrlControlAdapter
from grpo_guard.schema.events import SyncEvent, event_from_payload


def _mk_log(tmp_path):
    from grpo_guard.store.append_log import AppendLog

    log_ = AppendLog(tmp_path / "events", run_id="run-sync", lease_id="trl")
    epoch = log_.acquire_lease()
    control = TrlControlAdapter(log_, "run-sync", seq_provider=lambda: max(
        [e["lifecycle_seq"] for e in log_.iterate()] + [-1]) + 1)
    return log_, epoch, control


def _types(log_):
    return [e["event_type"] for e in log_.iterate()]


def test_sync_begin_then_complete_records_runtime_loaded_after_calls(tmp_path):
    log_, epoch, control = _mk_log(tmp_path)
    begin = control.sync_begin(1, "a" * 64, epoch, required_epoch=epoch)
    assert [e.event_type for e in begin] == ["sync_requested", "sync_started"]
    # runtime_loaded must NOT exist yet
    assert "runtime_loaded" not in _types(log_)

    loaded = control.sync_complete(1, "a" * 64, epoch, begin[0].sync_id,
                                   observed_sync_calls=398, param_digest="d" * 64,
                                   required_epoch=epoch)
    assert loaded.event_type == "runtime_loaded"
    assert "observed 398 update_named_param calls" in (loaded.status_detail or "")


def test_sync_failed_never_writes_runtime_loaded(tmp_path):
    log_, epoch, control = _mk_log(tmp_path)
    begin = control.sync_begin(1, "a" * 64, epoch, required_epoch=epoch)
    failed = control.sync_failed(1, "a" * 64, epoch, begin[0].sync_id,
                                 "RuntimeError: connection reset", required_epoch=epoch)
    assert failed.event_type == "sync_failed"
    types = _types(log_)
    assert "runtime_loaded" not in types
    assert "sync_failed" in types


def test_canary_mismatch_is_not_canary_passed(tmp_path):
    log_, epoch, control = _mk_log(tmp_path)
    begin = control.sync_begin(1, "a" * 64, epoch, required_epoch=epoch)
    control.sync_complete(1, "a" * 64, epoch, begin[0].sync_id, 398, "d" * 64, required_epoch=epoch)
    mm = control.canary_mismatch(1, "a" * 64, epoch, begin[0].sync_id,
                                 {"max_token_drift": 8}, required_epoch=epoch)
    assert mm.event_type == "canary_mismatch"
    types = _types(log_)
    assert "canary_passed" not in types
    assert "canary_mismatch" in types


class _NoopSyncStub:
    def __init__(self):
        self.calls = 0

    def update_named_param(self, name, param) -> None:
        self.calls += 1  # silently drop the weight


def test_noop_sync_stub_returns_success_but_drops_weights():
    stub = _NoopSyncStub()
    stub.update_named_param("model.layers.0.weight", object())  # returns None, no raise
    assert stub.calls == 1


def test_compare_sketches_detects_divergence():
    from examples.countdown.sync_noop_experiment import compare_sketches

    assert compare_sketches([[1, 2, 3]], [[1, 2, 3]]) == 0
    assert compare_sketches([[1, 2, 3]], [[1, 9, 3]]) == 1
    assert compare_sketches([[1, 2], [3, 4]], [[1, 2], [5, 6]]) == 2
