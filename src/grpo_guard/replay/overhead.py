"""Guard overhead measurement (design doc §13, §16.3).

Fixed workload, fixed artifact set, ≥3 short repeats.  Guard-ON = full
reason-coded validation of the real loop envelopes; Guard-OFF = the
consumption path without validation (the accident path).  Reports raw
values, mean and dispersion — never only the best sample.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from grpo_guard.day3 import load_loop_evidence, trajectory_from_loop
from grpo_guard.schema.events import GenerationEvent
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope


def measure_overhead(loop_dir: Path, repeats: int = 3, out_path: Path | None = None) -> dict:
    events, store, run_id = load_loop_evidence(loop_dir)
    gens = sorted(
        (e for e in events.values() if isinstance(e, GenerationEvent) and e.behavior_policy_version == 0),
        key=lambda e: e.lifecycle_seq,
    )[:8]  # fixed workload: first 8 real trajectories
    split = {"split_id": "split-train", "split_name": "train",
             "prompt_ids": sorted({g.prompt_id for g in gens})}
    protocol = ProtocolConfig(name="strict_v01", mode="strict_on_policy")
    trajectories = [
        trajectory_from_loop(events, store, run_id, g, g.checkpoint_manifest_sha256, split) for g in gens
    ]

    def guard_on():
        for t in trajectories:
            ctx = ValidationContext(
                envelope=t.envelope, store=t.store, events=t.events,
                policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
                protocol=protocol,
            )
            validate_envelope(ctx, "identity_pre_reward")

    def guard_off():
        # consumption without validation: re-read artifacts only
        for t in trajectories:
            for ref in (t.sequence_ref,):
                t.store.get(ref)

    on_times, off_times = [], []
    for i in range(repeats):
        t0 = time.perf_counter()
        guard_on()
        on_times.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        guard_off()
        off_times.append((time.perf_counter() - t0) * 1000.0)

    def stats(xs):
        return {"raw_ms": [round(v, 2) for v in xs],
                "mean_ms": round(statistics.mean(xs), 2),
                "stdev_ms": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0}

    result = {
        "workload": {"trajectories": len(trajectories), "repeats": repeats},
        "guard_on_ms": stats(on_times),
        "guard_off_ms": stats(off_times),
        "overhead_mean_ms": round(statistics.mean(on_times) - statistics.mean(off_times), 2),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def stage_timings(loop_dir: Path) -> dict:
    """Stage durations from event timestamps (design doc §13)."""
    events, _, _ = load_loop_evidence(loop_dir)
    times = {}
    for ev in events.values():
        ts = ev.created_at_utc
        times.setdefault(ev.event_type, []).append(ts)
    # coarse stages from the earliest event of each phase
    stage_of = {
        "sync": ["sync_requested", "canary_passed"],
        "rollout": ["generation_finished"],
        "validation": ["validation_decision"],
        "reward": ["reward_finished"],
        "update": ["update_committed"],
    }
    import datetime

    def parse(ts):
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))

    result = {}
    for stage, types in stage_of.items():
        stamps = [parse(t) for t in times.get(types[0], [])]
        if stamps:
            result[stage] = {"events": len(stamps), "first": str(stamps[0]), "last": str(stamps[-1])}
    return result
