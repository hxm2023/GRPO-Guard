"""contract-check: run frozen case directories through the validator
(design doc §15.4, §16.2 Day 3 gate inputs)."""

from __future__ import annotations

import json
from pathlib import Path

from grpo_guard.schema.decisions import ValidationDecision
from grpo_guard.schema.envelope import TrajectoryEnvelope
from grpo_guard.schema.events import EventBase, ValidationDecisionEvent, event_from_payload
from grpo_guard.schema.manifests import PolicyManifest, SplitManifest
from grpo_guard.store.artifact_store import ArtifactStore
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope


def load_case(case_dir: Path) -> tuple[dict, dict]:
    """Load a frozen case: (spec, context payloads)."""
    spec = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    inputs = case_dir / "inputs"
    envelope = TrajectoryEnvelope(**json.loads((inputs / "envelope.json").read_text(encoding="utf-8")))
    policy = PolicyManifest(**json.loads((inputs / "policy_manifest.json").read_text(encoding="utf-8")))
    split = SplitManifest(**json.loads((inputs / "split_manifest.json").read_text(encoding="utf-8")))
    events: dict[str, EventBase] = {}
    for path in sorted(inputs.glob("event_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        events[payload["event_id"]] = event_from_payload(payload)
    store = ArtifactStore(case_dir / "store")
    for blob in sorted(inputs.glob("artifact_*")):
        (store.blobs / blob.name.removeprefix("artifact_")).write_bytes(blob.read_bytes())
    bogus = None
    if (inputs / "bogus_sequence_ref.json").exists():
        from grpo_guard.schema.artifacts import ArtifactRef

        bogus = ArtifactRef(**json.loads((inputs / "bogus_sequence_ref.json").read_text(encoding="utf-8")))
    context = {}
    if (inputs / "context.json").exists():
        context = json.loads((inputs / "context.json").read_text(encoding="utf-8"))
    return spec, {"envelope": envelope, "policy_manifest": policy, "split_manifest": split,
                  "events": events, "store": store, "bogus_sequence_ref": bogus, "context": context}


def run_contract_check(cases_root: Path, out_dir: Path) -> dict:
    results = []
    passed = 0
    total = 0
    for case_dir in sorted(cases_root.iterdir()):
        if not case_dir.is_dir() or not (case_dir / "case.json").exists():
            continue
        spec, payload = load_case(case_dir)
        protocol = ProtocolConfig(name=spec.get("protocol", "strict_v01"), mode="strict_on_policy")
        ctx = ValidationContext(
            envelope=payload["envelope"], store=payload["store"], events=payload["events"],
            policy_manifest=payload["policy_manifest"], split_manifest=payload["split_manifest"],
            protocol=protocol,
        )
        if payload.get("context", {}).get("split_registry"):
            from grpo_guard.schema.manifests import SplitManifest

            ctx.split_registry = {
                name: SplitManifest(**data) for name, data in payload["context"]["split_registry"].items()
            }
        if payload.get("context", {}).get("eval_protocol_sha256"):
            ctx.eval_protocol_sha256 = payload["context"]["eval_protocol_sha256"]
        if payload.get("context", {}).get("reward_verifier_registry"):
            ctx.reward_verifier_registry = payload["context"]["reward_verifier_registry"]
        if payload.get("context", {}).get("requires_update_input"):
            from grpo_guard.schema.artifacts import EventRef
            from grpo_guard.schema.events import UpdateInputEvent

            gen = payload["events"][payload["envelope"].generation_event.event_id]
            ctx.update_input_event = UpdateInputEvent(
                event_id=f"uinput-{payload['envelope'].envelope_id}",
                run_id=payload["envelope"].run_id, component_id="materializer",
                lifecycle_seq=gen.lifecycle_seq + 100, created_at_utc=gen.created_at_utc,
                update_id="update-1", preupdate_envelope=payload["envelope"].ref(),
                preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
                sequence_token_ids=gen.sequence_token_ids, loss_mask=gen.loss_mask,
                authoritative_behavior_logprob_event=payload["envelope"].training_contract.authoritative_behavior_logprob_event,
                authoritative_behavior_logprobs=gen.service_behavior_logprobs,
                reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
                materialized_layout_sha256="0" * 64, single_use_nonce_sha256="0" * 64,
                tokenizer_called=False,
            ).seal()
        if payload.get("bogus_sequence_ref") is not None:
            from grpo_guard.schema.artifacts import EventRef
            from grpo_guard.schema.events import UpdateInputEvent

            gen = payload["events"][payload["envelope"].generation_event.event_id]
            ctx.update_input_event = UpdateInputEvent(
                event_id=f"uinput-{payload['envelope'].envelope_id}",
                run_id=payload["envelope"].run_id, component_id="materializer",
                lifecycle_seq=gen.lifecycle_seq + 1, created_at_utc=payload["events"][gen.event_id].created_at_utc,
                update_id="update-1",
                preupdate_envelope=payload["envelope"].ref(),
                preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
                sequence_token_ids=payload["bogus_sequence_ref"],
                loss_mask=gen.loss_mask,
                authoritative_behavior_logprob_event=payload["envelope"].training_contract.authoritative_behavior_logprob_event,
                authoritative_behavior_logprobs=gen.service_behavior_logprobs,
                reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
                materialized_layout_sha256="0" * 64,
                single_use_nonce_sha256="0" * 64,
                tokenizer_called=False,
            ).seal()
        needs_full = (
            payload["envelope"].envelope_stage == "pre_update"
            or payload.get("bogus_sequence_ref") is not None
            or bool(payload.get("context", {}).get("split_registry"))
            or bool(payload.get("context", {}).get("eval_protocol_sha256"))
            or bool(payload.get("context", {}).get("reward_verifier_registry"))
            or bool(payload.get("context", {}).get("requires_update_input"))
            or bool(payload["split_manifest"].content_sha256s)  # F10: D004 is a pre-update rule
        )
        stage = "full_pre_update" if needs_full else "identity_pre_reward"
        decision = validate_envelope(ctx, stage).decision_payload
        expected = spec["expected_decision"]
        required = set(spec.get("required_reason_codes", []))
        match = decision.decision == expected and required.issubset(set(decision.reason_codes))
        total += 1
        passed += 1 if match else 0
        results.append({
            "case_id": spec["case_id"],
            "expected_decision": expected,
            "decision": decision.decision,
            "reason_codes": decision.reason_codes,
            "match": match,
        })

    manifest = {
        "command": "contract-check",
        "cases_root": str(cases_root),
        "results": results,
        "summary": {"total": total, "passed": passed},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "contract_check.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
