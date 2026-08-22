"""Day 3: F1-F4 fault matrix over the REAL closed-loop artifacts
(design doc §11, §16.2 Correctness Gate).

The loop's append-only events and content-addressed artifacts are replayed
into ValidationContexts; the four canonical injectors mutate exactly one
field of the real producer events; every decision is reason-coded and
machine-readable.  The gate compares against the PRE-REGISTERED
expectations in configs/faults/f1_f4_v01.yaml — no post-hoc edits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from grpo_guard import testing
from grpo_guard.faults import (
    inject_f1_stale_sync,
    inject_f1_static_rollout,
    inject_f2_misbound_logprob,
    inject_f2_wrong_generation,
    inject_f3_retokenization,
    inject_f3_retokenized_sequence,
    inject_f3_template_variant,
    inject_f4_mask_shift,
)
from grpo_guard.schema.artifacts import EventRef, ManifestRef
from grpo_guard.schema.envelope import TrajectoryEnvelope, TrainingContract
from grpo_guard.schema.events import GenerationEvent, SyncEvent, event_from_payload
from grpo_guard.schema.manifests import PolicyManifest, SplitManifest
from grpo_guard.store.artifact_store import ArtifactStore
from grpo_guard.store.canonical_json import canonical_sha256
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope


def load_loop_evidence(loop_dir: Path) -> tuple[dict, ArtifactStore, str]:
    """Load events + store from the Day 2 loop output."""
    events = {}
    run_id = ""
    for path in sorted((loop_dir / "events").rglob("*.json")):
        if "edges" in path.parts or path.name == "lease.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        events[payload["event_id"]] = event_from_payload(payload)
        run_id = payload["run_id"]
    store = ArtifactStore(loop_dir / "store")
    return events, store, run_id


def build_envelope(run_id, gen: GenerationEvent, ckpt_sha: str, split, stage="pre_reward"):
    return TrajectoryEnvelope(
        envelope_id=f"env-{gen.event_id}-{stage}",
        envelope_stage=stage, run_id=run_id, request_id=gen.request_id,
        generation_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
        scoring_event=None, reward_event=None,
        policy_manifest=ManifestRef(uri="", manifest_id="pm-0", sha256=ckpt_sha),
        split_manifest=ManifestRef(uri="", manifest_id=split["split_id"], sha256=canonical_sha256(split)),
        parent_envelope_sha256=None, parent_identity_decision=None,
        training_contract=TrainingContract(
            protocol="strict_on_policy", trainer_parent_policy_version=0,
            consuming_update_id="update-1", max_policy_lag_versions=0,
            behavior_logprob_source="generation_service",
            authoritative_behavior_logprob_event=EventRef(uri="", event_id=gen.event_id, event_sha256=gen.event_sha256),
            diagnostic_non_authoritative_logprobs_allowed=False,
        ),
    ).seal()


def trajectory_from_loop(events, store, run_id, gen: GenerationEvent, ckpt_sha: str, split) -> testing.Trajectory:
    seq = np.frombuffer(store.get(gen.sequence_token_ids), dtype=np.int32).copy()
    target = np.frombuffer(store.get(gen.completion_target_mask), dtype=np.int8).copy()
    loss = np.frombuffer(store.get(gen.loss_mask), dtype=np.int8).copy()
    lp = (
        np.frombuffer(store.get(gen.service_behavior_logprobs), dtype=np.float32).copy()
        if gen.service_behavior_logprobs else np.zeros(0, dtype=np.float32)
    )
    envelope = build_envelope(run_id, gen, ckpt_sha, split)
    return testing.Trajectory(
        run_id=run_id, events=events,
        policy_manifest=PolicyManifest(
            manifest_id="pm-0", model_id="Qwen/Qwen3-4B", model_revision="r",
            policy_version=0, weights=[], checkpoint_manifest_sha256=ckpt_sha,
            tokenizer_sha256=gen.tokenizer_sha256, chat_template_sha256=gen.chat_template_sha256,
            precision="bf16", adapter_kind="full", code_commit_sha="c", config_sha256="s",
        ),
        split_manifest=SplitManifest(**split),
        envelope=envelope, store=store, sequence=seq, target_mask=target, loss_mask=loss,
        logprobs=lp, completion_text="", goal=0, target_numbers=[], reward_components={},
        sequence_ref=gen.sequence_token_ids,
        sync_events=sorted((e for e in events.values() if isinstance(e, SyncEvent)), key=lambda e: e.lifecycle_seq),
    )


INJECTORS = {
    "f1_static_rollout": lambda t, v: (
        inject_f1_stale_sync(t) if v.get("kind") == "stale_sync"
        else inject_f1_static_rollout(t, v["runtime_version"], v["claimed_parent"])
    ),
    "f2_misbound_logprob": lambda t, v: (
        inject_f2_wrong_generation(t) if v.get("kind") == "wrong_generation"
        else inject_f2_misbound_logprob(t, v["scorer_policy_version"])
    ),
    "f3_retokenization": lambda t, v: (
        inject_f3_retokenized_sequence(t, "b" * 64) if v.get("kind") == "sequence"
        else (inject_f3_template_variant(t) if v.get("kind") == "template" else inject_f3_retokenization(t))
    ),
    "f4_mask_shift": lambda t, v: inject_f4_mask_shift(t, v["shift"]),
}


def run_day3_matrix(loop_dir: Path, config_path: Path, out_dir: Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    events, store, run_id = load_loop_evidence(loop_dir)
    protocol = ProtocolConfig(name=cfg.get("protocol", "strict_v01"), mode="strict_on_policy")

    # the real v0 generation events are the normal set
    gens = sorted(
        (e for e in events.values() if isinstance(e, GenerationEvent) and e.behavior_policy_version == 0),
        key=lambda e: e.lifecycle_seq,
    )
    if len(gens) < 8:
        raise RuntimeError(f"loop evidence has only {len(gens)} v0 generations; need >= 8")

    split = {"split_id": "split-train", "split_name": "train",
             "prompt_ids": sorted({g.prompt_id for g in gens})}

    def validate(t: testing.Trajectory):
        ctx = ValidationContext(
            envelope=t.envelope, store=t.store, events=t.events,
            policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
            protocol=protocol,
        )
        stage = "identity_pre_reward"
        if getattr(t, "bogus_sequence_ref", None) is not None:
            from grpo_guard.schema.events import UpdateInputEvent

            gen = t.events[t.envelope.generation_event.event_id]
            ctx.update_input_event = UpdateInputEvent(
                event_id=f"uinput-{t.envelope.envelope_id}", run_id=t.run_id,
                component_id="materializer", lifecycle_seq=t.next_seq(),
                created_at_utc=t.events[gen.event_id].created_at_utc,
                update_id="update-1", preupdate_envelope=t.envelope.ref(),
                preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
                sequence_token_ids=t.bogus_sequence_ref, loss_mask=gen.loss_mask,
                authoritative_behavior_logprob_event=t.envelope.training_contract.authoritative_behavior_logprob_event,
                authoritative_behavior_logprobs=gen.service_behavior_logprobs,
                reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
                materialized_layout_sha256="0" * 64, single_use_nonce_sha256="0" * 64,
                tokenizer_called=False,
            ).seal()
            stage = "full_pre_update"
        return validate_envelope(ctx, stage).decision_payload

    # ---- normal set: every real v0 trajectory must stay ALLOW --------------
    normals = []
    for i, gen in enumerate(gens):
        t = trajectory_from_loop(events, store, run_id, gen, gens[0].checkpoint_manifest_sha256, split)
        d = validate(t)
        normals.append({"case_id": f"normal_{i:02d}", "generation": gen.event_id,
                        "decision": d.decision, "reason_codes": d.reason_codes})

    # ---- canonical + held-out faults on the REAL first generation ----------
    # each variant gets a FRESH trajectory built from the real events — the
    # injectors mutate their input, and sharing one base would contaminate
    # later variants (review finding C3)
    results = []
    for case in cfg["cases"]:
        injector = INJECTORS[case["fault"]]
        for variant in case["variants"]:
            base = trajectory_from_loop(events, store, run_id, gens[0], gens[0].checkpoint_manifest_sha256, split)
            ft = injector(base, variant)
            d = validate(ft)
            expected = variant.get("expected_decision", case["expected_decision"])
            required = variant.get("required_reason_codes", case["required_reason_codes"])
            match = d.decision == expected and set(required).issubset(set(d.reason_codes))
            results.append({
                "case_id": f"{case['id']}:{variant['name']}",
                "fault": case["fault"], "variant": variant,
                "expected_decision": expected,
                "required_reason_codes": required,
                "decision": d.decision, "reason_codes": d.reason_codes,
                "match": match,
            })

    # ---- boundary cases (pre-registered in the faults config, §16.2) --------
    boundaries = []
    for spec in cfg.get("boundary_cases", []):
        t = testing.build_trajectory(policy_version=0, **spec.get("kwargs", {}))
        d = validate(t)
        codes = spec.get("required_reason_codes", [])
        match = d.decision == spec["expected_decision"] and (not codes or set(codes).issubset(set(d.reason_codes)))
        boundaries.append({"case_id": spec["id"], "expected_decision": spec["expected_decision"],
                           "decision": d.decision, "reason_codes": d.reason_codes, "match": match})

    canonical = [r for r in results if r["variant"].get("name") == "canonical"]
    normal_ok = all(n["decision"] == "allow" for n in normals)
    # strict stale acceptance: any fault-injected trajectory that got ALLOW
    stale_accept = sum(1 for r in results if r["decision"] == "allow")
    summary = {
        "canonical_matched": sum(1 for r in canonical if r["match"]),
        "canonical_total": len(canonical),
        "fault_matrix_matched": sum(1 for r in results if r["match"]),
        "fault_matrix_total": len(results),
        "normal_allow": sum(1 for n in normals if n["decision"] == "allow"),
        "normal_total": len(normals),
        "normal_quarantine": sum(1 for n in normals if n["decision"] == "quarantine"),
        "normal_reject": sum(1 for n in normals if n["decision"] == "reject"),
        "boundary_matched": sum(1 for b in boundaries if b["match"]),
        "boundary_total": len(boundaries),
        "strict_stale_acceptance": stale_accept,
        "gate_pass": (
            sum(1 for r in canonical if r["match"]) == len(canonical)
            and normal_ok
            and len(normals) >= 8
            and all(b["match"] for b in boundaries)
        ),
    }

    matrix = {
        "matrix_id": cfg.get("matrix_id", "f1_f4_v01"),
        "protocol": "strict_on_policy",
        "source": {"loop_dir": str(loop_dir), "generations_used": [g.event_id for g in gens[:4]]},
        "results": results,
        "normal": normals,
        "boundary": boundaries,
        "summary": summary,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fault_matrix.json").write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    return matrix
