"""Training recovery plan tests (resume from event log)."""

from __future__ import annotations

import json
from pathlib import Path

from grpo_guard.resume import analyze_training, write_recovery_plan
from grpo_guard.schema.events import TrainingStepEvent, UpdateEvent
from grpo_guard.schema.artifacts import EventRef


def _mk_events(tmp_path: Path, n_steps: int = 3) -> Path:
    d = tmp_path / "events"
    d.mkdir()
    for k in range(1, n_steps + 1):
        ev = TrainingStepEvent(
            event_id=f"tstep-{k}", event_type="training_step_finished", run_id="run-rl",
            component_id="grpo_trainer", lifecycle_seq=k * 10,
            created_at_utc="t",
            update_id=f"update-{k}", parent_policy_version=k - 1, output_policy_version=k,
            rollout_sequences=32, consumed_sequences=32, success_rate=0.5 + k * 0.05,
            loss=-0.001 * k, ratio_p50=1.1, ratio_max=2.0, weight_delta_fp32_vs_v0=k * 2.0,
        ).seal()
        (d / f"{ev.event_id}.json").write_text(json.dumps(ev.model_dump(mode="json")), encoding="utf-8")
        upd = UpdateEvent(
            event_id=f"upd-{k}", event_type="update_committed", run_id="run-rl",
            component_id="trl_control", lifecycle_seq=k * 10 + 1, created_at_utc="t",
            update_id=f"update-{k}", transaction_id=f"txn-{k}", lease_epoch=1,
            idempotency_key=f"upd-{k}", parent_policy_version=k - 1, output_policy_version=k,
            input_preupdate_envelope_sha256s=[], checkpoint_manifest_sha256=f"{k:064d}",
            update_input_event=EventRef(uri="", event_id="u-1", event_sha256="0" * 64),
        ).seal()
        (d / f"{upd.event_id}.json").write_text(json.dumps(upd.model_dump(mode="json")), encoding="utf-8")
    return d


def test_recovery_plan_from_event_log(tmp_path: Path):
    events = _mk_events(tmp_path, n_steps=3)
    plan = analyze_training(events)
    assert plan.ok
    assert plan.last_step == 3
    assert plan.next_step == 4
    assert plan.checkpoint_dir == "ckpt_v3"
    assert plan.checkpoint_manifest_sha256 == f"{3:064d}"
    assert len(plan.steps) == 3
    assert plan.steps[-1]["success_rate"] == 0.65


def test_recovery_plan_empty_log(tmp_path: Path):
    d = tmp_path / "events"
    d.mkdir()
    plan = analyze_training(d)
    assert not plan.ok
    assert "nothing to resume" in plan.failures[0]


def test_write_recovery_plan(tmp_path: Path):
    events = _mk_events(tmp_path, n_steps=2)
    out = tmp_path / "plan.json"
    plan = write_recovery_plan(events, out)
    assert out.exists()
    assert plan["last_step"] == 2 and plan["next_step"] == 3
