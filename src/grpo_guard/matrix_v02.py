"""v0.2-preview F5-F8 matrix over the REAL loop artifacts (design doc §11).

Runs the F5-F8 injectors against the committed closed-loop generation
events; decisions are reason-coded and compared to the pre-registered
expectations in configs/faults/f5_f8_v02.yaml.  These are v0.2-preview
fixtures — the v0.1 matrix (f1_f4_v01) is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from grpo_guard import testing
from grpo_guard.day3 import (
    INJECTORS as _F1F4,  # noqa: F401 (kept for interface parity)
    build_envelope,
    load_loop_evidence,
    trajectory_from_loop,
)
from grpo_guard.schema.artifacts import EventRef
from grpo_guard.faults.f5_f8 import (
    inject_f5_split_leakage,
    inject_f6_evaluator_alias,
    inject_f6_no_alias,
    inject_f7_event_reorder,
    inject_f7_sync_reorder,
    inject_f8_artifact_mutation,
)
from grpo_guard.faults.f9_f10 import inject_f9_reward_hacking, inject_f10_data_poisoning
from grpo_guard.schema.events import GenerationEvent, RewardEvent
from grpo_guard.schema.manifests import SplitManifest
from grpo_guard.store.canonical_json import canonical_sha256
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope

def _resolve_eval_protocol(variant: dict, declared_sha: str | None) -> str:
    spec = variant.get("eval_protocol", "declared")
    if spec == "declared":
        if declared_sha is None:
            raise RuntimeError("f6 requires a declared eval protocol (none in loop evidence)")
        return declared_sha
    return spec


def _f6_inject(t, v):
    kind = v.get("kind", "alias")
    if kind == "no_alias":
        return inject_f6_no_alias(t, v["eval_protocol"])
    return inject_f6_evaluator_alias(t, v["eval_protocol"],
                                     eval_reward_version=v.get("eval_reward_version"))


def _f7_inject(t, v):
    if v.get("kind") == "sync_late":
        return inject_f7_sync_reorder(t)
    return inject_f7_event_reorder(t, scorer_policy_version=v.get("scorer_policy_version"))


INJECTORS_V02 = {
    "f5_split_leakage": lambda t, v: inject_f5_split_leakage(
        t, v.get("other_split", "held_out"),
        prompt_id=v.get("prompt_id"), extra_split=v.get("extra_split")),
    "f6_evaluator_alias": _f6_inject,
    "f7_event_reorder": _f7_inject,
    "f8_artifact_mutation": lambda t, v: inject_f8_artifact_mutation(t, v.get("which", "sequence")),
    "f9_reward_hacking": lambda t, v: (
        inject_f9_reward_hacking(t, fake_version=v.get("fake_version", "fake-reward-v1"),
                                 wrong_protocol="f" * 64 if v.get("wrong_protocol") else None)
    ),
    "f10_data_poisoning": lambda t, v: inject_f10_data_poisoning(t),
}


def pre_update_base_for(events, store, run_id, split, gen) -> testing.Trajectory:
    """F6 needs a pre-update envelope WITH a reward event (R006 fires on
    the reward); build it from the loop's real reward for this generation
    and a synthetic ALLOW parent identity."""
    from grpo_guard.schema.decisions import ValidationDecision
    from grpo_guard.schema.events import ValidationDecisionEvent

    base = trajectory_from_loop(events, store, run_id, gen, gen.checkpoint_manifest_sha256, split)
    reward = next(
        (e for e in events.values()
         if isinstance(e, RewardEvent) and e.source_generation_event.event_id == gen.event_id),
        None,
    )
    if reward is None:
        raise RuntimeError("F6 needs a reward event in loop evidence")
    parent = ValidationDecisionEvent(
        event_id=f"vdec-{base.envelope.envelope_id}-parent",
        event_type="validation_decision", run_id=run_id, component_id="validator",
        lifecycle_seq=gen.lifecycle_seq + 1, created_at_utc=testing.now_utc(),
        decision_payload=ValidationDecision(decision="allow", validation_stage="identity_pre_reward",
                                            reason_codes=["G001_POLICY_MATCH"]),
    ).seal()
    env = build_envelope(run_id, gen, gen.checkpoint_manifest_sha256, split, stage="pre_update")
    env = env.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{env.envelope_id}-f6",
        "reward_event": EventRef(uri="", event_id=reward.event_id, event_sha256=reward.event_sha256),
        "parent_identity_decision": EventRef(uri="", event_id=parent.event_id, event_sha256=parent.event_sha256),
    })
    env.envelope_sha256 = ""
    env = env.seal()
    base.events[reward.event_id] = reward
    base.events[parent.event_id] = parent
    base.envelope = env
    return base


def _eval_protocol_sha(events: dict) -> str | None:
    """The eval protocol: declared as the 'calibration' reward protocol, or a
    DISTINCT eval protocol identity when the loop carries no eval reward
    events (must never equal the training protocol, or the no-alias
    boundary variant could not be constructed)."""
    import hashlib

    eval_rewards = [e for e in events.values()
                    if isinstance(e, RewardEvent) and e.reward_version.endswith("eval")]
    if eval_rewards:
        return eval_rewards[0].evaluator_protocol_sha256
    return hashlib.sha256(b"countdown-eval-protocol-v1").hexdigest()


def run_v02_matrix(loop_dir: Path, config_path: Path, out_dir: Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    events, store, run_id = load_loop_evidence(loop_dir)
    protocol = ProtocolConfig(name=cfg.get("protocol", "strict_v01"), mode="strict_on_policy")
    eval_proto = _eval_protocol_sha(events)

    gens = sorted(
        (e for e in events.values() if isinstance(e, GenerationEvent) and e.behavior_policy_version == 0),
        key=lambda e: e.lifecycle_seq,
    )
    if len(gens) < 4:
        raise RuntimeError(f"loop evidence has only {len(gens)} v0 generations; need >= 4")

    split = {"split_id": "split-train", "split_name": "train",
             "prompt_ids": sorted({g.prompt_id for g in gens})}
    held = {"split_id": "split-held_out", "split_name": "held_out",
            "prompt_ids": [gens[0].prompt_id]}  # the F5 fixture leaks this one

    from grpo_guard.schema.events import UpdateInputEvent

    first_update_input = next(
        (e for e in events.values() if isinstance(e, UpdateInputEvent)),
        None,
    )

    def update_input_for(t: testing.Trajectory) -> UpdateInputEvent | None:
        """Minimal update input for THIS trajectory, carrying the real
        consuming-update lifecycle seq (so L005 can compare ordering) but
        referencing THIS generation's artifacts (no spurious T004/L008)."""
        if first_update_input is None:
            return None
        gen = t.events[t.envelope.generation_event.event_id]
        return UpdateInputEvent(
            event_id=f"uinput-{t.envelope.envelope_id}", run_id=t.run_id,
            component_id="materializer",
            lifecycle_seq=first_update_input.lifecycle_seq,
            created_at_utc=gen.created_at_utc,
            update_id="update-1", preupdate_envelope=t.envelope.ref(),
            preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
            sequence_token_ids=gen.sequence_token_ids, loss_mask=gen.loss_mask,
            authoritative_behavior_logprob_event=t.envelope.training_contract.authoritative_behavior_logprob_event,
            authoritative_behavior_logprobs=gen.service_behavior_logprobs,
            reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
            materialized_layout_sha256="0" * 64, single_use_nonce_sha256="0" * 64,
            tokenizer_called=False,
        ).seal()

    def validate(t: testing.Trajectory, with_update_input: bool = False):
        ctx = ValidationContext(
            envelope=t.envelope, store=t.store, events=t.events,
            policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
            protocol=protocol,
            split_registry=getattr(t, "split_registry", None),
            eval_protocol_sha256=getattr(t, "eval_protocol_sha256", None),
            reward_verifier_registry=getattr(t, "reward_verifier_registry", None),
            update_input_event=update_input_for(t) if with_update_input else None,
        )
        return validate_envelope(ctx, "full_pre_update").decision_payload

    results = []
    for case in cfg["cases"]:
        injector = INJECTORS_V02[case["fault"]]
        for variant in case["variants"]:
            base = trajectory_from_loop(events, store, run_id, gens[0], gens[0].checkpoint_manifest_sha256, split)
            if case["fault"] in ("f6_evaluator_alias", "f9_reward_hacking"):
                if case["fault"] == "f6_evaluator_alias":
                    declared = _resolve_eval_protocol(variant, eval_proto)
                    variant = {**variant, "eval_protocol": declared}
                base = pre_update_base_for(events, store, run_id, split, gens[0])
            if case["fault"] == "f8_artifact_mutation":
                # F8 mutates blob bytes — run it against an ISOLATED store
                # clone so the shared loop evidence is never touched
                import shutil
                import tempfile

                clone = Path(tempfile.mkdtemp(prefix="grpo-guard-f8-"))
                shutil.copytree(store.root, clone, dirs_exist_ok=True)
                from grpo_guard.store.artifact_store import ArtifactStore

                base = testing.Trajectory(
                    run_id=base.run_id, events=base.events,
                    policy_manifest=base.policy_manifest, split_manifest=base.split_manifest,
                    envelope=base.envelope, store=ArtifactStore(clone),
                    sequence=base.sequence, target_mask=base.target_mask, loss_mask=base.loss_mask,
                    logprobs=base.logprobs, completion_text=base.completion_text, goal=base.goal,
                    target_numbers=base.target_numbers, reward_components=base.reward_components,
                    sync_events=base.sync_events, sequence_ref=base.sequence_ref,
                )
            ft = injector(base, variant)
            need_update_input = (
                case["fault"] == "f7_event_reorder" and variant.get("kind", "scoring_late") == "scoring_late"
            )
            d = validate(ft, with_update_input=need_update_input)
            required = variant.get("required_reason_codes", case["required_reason_codes"])
            expected = variant.get("expected_decision", case["expected_decision"])
            match = d.decision == expected and set(required).issubset(set(d.reason_codes))
            results.append({
                "case_id": f"{case['id']}:{variant['name']}",
                "fault": case["fault"], "variant": variant,
                "expected_decision": expected, "required_reason_codes": required,
                "decision": d.decision, "reason_codes": d.reason_codes, "match": match,
            })

    normals = []
    for i in range(cfg["normal_cases"]["count"]):
        base = trajectory_from_loop(events, store, run_id, gens[i], gens[0].checkpoint_manifest_sha256, split)
        d = validate(base)
        normals.append({"case_id": f"normal_{i}", "decision": d.decision, "reason_codes": d.reason_codes})

    summary = {
        "matched": sum(1 for r in results if r["match"]),
        "total": len(results),
        "normal_allow": sum(1 for n in normals if n["decision"] == "allow"),
        "normal_total": len(normals),
        "gate_pass": all(r["match"] for r in results) and all(n["decision"] == "allow" for n in normals),
    }
    matrix = {
        "matrix_id": cfg.get("matrix_id", "f5_f8_v02"),
        "protocol": "strict_on_policy",
        "scope": "v0.2-preview — NOT part of the v0.1 matrix (design doc §11)",
        "source": {"loop_dir": str(loop_dir).replace("\\", "/")},
        "results": results,
        "normal": normals,
        "summary": summary,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fault_matrix.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    return matrix


def freeze_v02_cases(loop_dir: Path, config_path: Path, freeze_root: Path) -> int:
    """Freeze the v0.2 variant cases (no-overwrite, design doc §15.4)."""
    from grpo_guard.frozen import write_case
    from grpo_guard.schema.decisions import ValidationDecision
    from grpo_guard.schema.events import ValidationDecisionEvent
    from grpo_guard.day3 import load_loop_evidence

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    events, store, run_id = load_loop_evidence(loop_dir)
    gens = sorted(
        (e for e in events.values() if isinstance(e, GenerationEvent) and e.behavior_policy_version == 0),
        key=lambda e: e.lifecycle_seq,
    )
    split = {"split_id": "split-train", "split_name": "train",
             "prompt_ids": sorted({g.prompt_id for g in gens})}
    written = 0
    for case in cfg["cases"]:
        injector = INJECTORS_V02[case["fault"]]
        for variant in case["variants"]:
            base = trajectory_from_loop(events, store, run_id, gens[0], gens[0].checkpoint_manifest_sha256, split)
            if case["fault"] == "f6_evaluator_alias":
                declared = _resolve_eval_protocol(variant, _eval_protocol_sha(events))
                variant = {**variant, "eval_protocol": declared}
                base = pre_update_base_for(events, store, run_id, split, gens[0])
            if case["fault"] == "f9_reward_hacking":
                base = pre_update_base_for(events, store, run_id, split, gens[0])
            if case["fault"] == "f8_artifact_mutation":
                import shutil
                import tempfile
                from grpo_guard.store.artifact_store import ArtifactStore

                clone = Path(tempfile.mkdtemp(prefix="grpo-guard-f8-freeze-"))
                shutil.copytree(store.root, clone, dirs_exist_ok=True)
                base = testing.Trajectory(
                    run_id=base.run_id, events=base.events,
                    policy_manifest=base.policy_manifest, split_manifest=base.split_manifest,
                    envelope=base.envelope, store=ArtifactStore(clone),
                    sequence=base.sequence, target_mask=base.target_mask, loss_mask=base.loss_mask,
                    logprobs=base.logprobs, completion_text=base.completion_text, goal=base.goal,
                    target_numbers=base.target_numbers, reward_components=base.reward_components,
                    sync_events=base.sync_events, sequence_ref=base.sequence_ref,
                )
            ft = injector(base, variant)
            expected = variant.get("expected_decision", case["expected_decision"])
            required = variant.get("required_reason_codes", case["required_reason_codes"])
            write_case(
                freeze_root, f"{case['id']}_{variant['name']}",
                expected, required, ft, notes=json.dumps(variant),
            )
            written += 1
    for i in range(cfg["normal_cases"]["count"]):
        base = trajectory_from_loop(events, store, run_id, gens[i], gens[0].checkpoint_manifest_sha256, split)
        write_case(freeze_root, f"normal_{i}", "allow", [], base, notes="v0.2 normal neighbor")
        written += 1
    return written
