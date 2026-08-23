"""Training recovery analysis (RL infra: resume from event log + checkpoints).

``grpo-guard resume`` reads an event log and produces a recovery plan:
the last completed training step (from ``training_step_finished`` events),
the checkpoint manifest to load, the per-step metrics recorded, and the
next step to run.  Training itself resumes by loading that checkpoint and
continuing the loop (``rl_training_loop.py --resume``).

This is the operational answer to "what happens when long training
dies?" — the append-only event stream is the source of truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from grpo_guard.schema.events import TrainingStepEvent, event_from_payload


@dataclass
class RecoveryPlan:
    ok: bool = True
    failures: list[str] = field(default_factory=list)
    last_step: int | None = None
    next_step: int | None = None
    checkpoint_manifest_sha256: str | None = None
    checkpoint_dir: str | None = None
    steps: list[dict] = field(default_factory=list)
    run_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "failures": self.failures,
            "last_step": self.last_step, "next_step": self.next_step,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_dir": self.checkpoint_dir,
            "run_id": self.run_id,
            "steps": self.steps,
            "resume_command": (f"rl_training_loop.py --resume (continues at step {self.next_step})"
                               if self.next_step else None),
        }


def analyze_training(events_dir: Path) -> RecoveryPlan:
    """Build a recovery plan from the event log alone."""
    plan = RecoveryPlan()
    by_step: dict[int, dict] = {}
    committed_sha: dict[int, str] = {}

    for p in sorted(events_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            ev = event_from_payload(payload)
        except (OSError, ValueError) as exc:
            plan.failures.append(f"{p.name}: unreadable ({exc})")
            continue
        if isinstance(ev, TrainingStepEvent):
            by_step[ev.output_policy_version] = {
                "step": ev.output_policy_version,
                "update_id": ev.update_id,
                "parent_policy_version": ev.parent_policy_version,
                "rollout_sequences": ev.rollout_sequences,
                "consumed_sequences": ev.consumed_sequences,
                "success_rate": ev.success_rate,
                "loss": ev.loss,
                "ratio_p50": ev.ratio_p50,
                "ratio_max": ev.ratio_max,
                "clip_fraction": ev.clip_fraction,
                "weight_delta_fp32_vs_v0": ev.weight_delta_fp32_vs_v0,
                "event_id": ev.event_id,
                "event_sha256": ev.event_sha256,
            }
            plan.run_id = plan.run_id or ev.run_id
        elif ev.event_type == "update_committed":
            committed_sha[ev.output_policy_version] = ev.checkpoint_manifest_sha256

    if not by_step:
        plan.ok = False
        plan.failures.append("no training_step_finished events found (nothing to resume)")
        return plan

    plan.steps = [by_step[k] for k in sorted(by_step)]
    plan.last_step = plan.steps[-1]["step"]
    plan.next_step = plan.last_step + 1
    plan.checkpoint_dir = f"ckpt_v{plan.last_step}"
    plan.checkpoint_manifest_sha256 = committed_sha.get(plan.last_step)
    return plan


def write_recovery_plan(events_dir: Path, out: Path, checkpoint_manifest_sha256: str | None = None) -> dict:
    plan = analyze_training(events_dir)
    plan.checkpoint_manifest_sha256 = checkpoint_manifest_sha256
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return plan.to_dict()
