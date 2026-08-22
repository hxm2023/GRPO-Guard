"""Fault-matrix runner (design doc §11, §16.2).

Runs the frozen matrix config: canonical faults, normal neighbors, boundary
and held-out variants — all against the pre-registered expected decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from grpo_guard import testing
from grpo_guard.day3 import INJECTORS
from grpo_guard.frozen import write_case
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope


def run_fault_matrix(
    config_path: Path,
    out_dir: Path,
    freeze_dir: Path | None = None,
    guard_mode: str = "strict_on_policy",
) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    protocol = ProtocolConfig(
        name=cfg.get("protocol", "strict_v01"),
        mode=guard_mode,
        max_policy_lag_versions=0 if guard_mode == "strict_on_policy" else 2,
        importance_correction=None if guard_mode == "strict_on_policy" else "importance-ratio-v1",
    )

    results: list[dict] = []
    passes = 0
    total = 0

    for case in cfg["cases"]:
        injector = INJECTORS[case["fault"]]
        for variant in case["variants"]:
            t = testing.build_trajectory(policy_version=0)
            ft = injector(t, variant)
            decision, stage = _validate_with_stage(ft, protocol)
            expected = variant.get("expected_decision", case["expected_decision"])
            required = variant.get("required_reason_codes", case["required_reason_codes"])
            match = (
                decision.decision == expected
                and set(required).issubset(set(decision.reason_codes))
            )
            total += 1
            passes += 1 if match else 0
            results.append({
                "case_id": f"{case['id']}:{variant['name']}",
                "expected_decision": expected,
                "decision": decision.decision,
                "reason_codes": decision.reason_codes,
                "required_reason_codes": required,
                "match": match,
            })
            if freeze_dir is not None:
                write_case(
                    freeze_dir,
                    f"{case['id']}_{variant['name']}",
                    expected,
                    required,
                    ft,
                    notes=json.dumps(variant),
                )

    # normal set: ≥8 cases, all allow, 0 quarantine, 0 reject
    normals = []
    for i in range(cfg["normal_cases"]["count"]):
        t = testing.build_trajectory(policy_version=0, run_id=f"run-normal-{i}")
        decision, _ = _validate_with_stage(t, protocol)
        normals.append({"case_id": f"normal_{i}", "decision": decision.decision, "reason_codes": decision.reason_codes})
        total += 1
        passes += 1 if decision.decision == "allow" else 0
        if freeze_dir is not None:
            write_case(freeze_dir, f"normal_{i}", "allow", [], t, notes="normal neighbor")

    # boundary cases (pre-registered in the faults config, §16.2)
    boundaries = []
    for spec in cfg.get("boundary_cases", []):
        case_id = spec["id"]
        codes = spec.get("required_reason_codes", [])
        expected = spec["expected_decision"]
        t = testing.build_trajectory(policy_version=0, run_id=f"run-{case_id}", **spec.get("kwargs", {}))
        decision, _ = _validate_with_stage(t, protocol)
        match = decision.decision == expected and (not codes or set(codes).issubset(set(decision.reason_codes)))
        total += 1
        passes += 1 if match else 0
        boundaries.append({
            "case_id": case_id, "expected_decision": expected, "decision": decision.decision,
            "reason_codes": decision.reason_codes, "match": match,
        })
        if freeze_dir is not None:
            write_case(freeze_dir, case_id, expected, codes, t)

    matrix = {
        "matrix_id": cfg.get("matrix_id", "f1_f4_v01"),
        "protocol": cfg.get("protocol", "strict_v01"),
        "guard_mode": guard_mode,
        "results": results,
        "normal": normals,
        "boundary": boundaries,
        "summary": {
            "total": total,
            "passed": passes,
            "canonical_faults": sum(1 for r in results if r["match"]),
            "canonical_total": len(results),
            "normal_allow_count": sum(1 for n in normals if n["decision"] == "allow"),
            "normal_total": len(normals),
            "normal_quarantine_count": sum(1 for n in normals if n["decision"] == "quarantine"),
            "normal_reject_count": sum(1 for n in normals if n["decision"] == "reject"),
            "strict_stale_acceptance": sum(1 for r in results if r["decision"] == "allow"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fault_matrix.json").write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return matrix


def _validate_with_stage(t: testing.Trajectory, protocol: ProtocolConfig):
    """Validate; F3 held-out (re-encoded sequence) needs the pre-update stage
    with an update_input_event referencing the bogus sequence (T004)."""
    from grpo_guard.schema.artifacts import EventRef
    from grpo_guard.schema.events import UpdateInputEvent

    ctx = ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=protocol,
    )
    stage = "identity_pre_reward"
    if getattr(t, "bogus_sequence_ref", None) is not None:
        gen = t.events[t.envelope.generation_event.event_id]
        upd = UpdateInputEvent(
            event_id=f"uinput-{t.envelope.envelope_id}",
            run_id=t.run_id, component_id="materializer",
            lifecycle_seq=t.next_seq(), created_at_utc=testing.now_utc(),
            update_id="update-1",
            preupdate_envelope=t.envelope.ref(),
            preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
            sequence_token_ids=t.bogus_sequence_ref,
            loss_mask=gen.loss_mask,
            authoritative_behavior_logprob_event=t.envelope.training_contract.authoritative_behavior_logprob_event,
            authoritative_behavior_logprobs=gen.service_behavior_logprobs,
            reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
            materialized_layout_sha256="0" * 64,
            single_use_nonce_sha256="0" * 64,
            tokenizer_called=False,
        ).seal()
        ctx.update_input_event = upd
        stage = "full_pre_update"
    return validate_envelope(ctx, stage).decision_payload, stage
