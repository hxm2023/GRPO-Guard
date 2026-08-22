"""GRPO-Guard 3-5 minute demo (design doc §23).

Shows, end to end and CPU-only:
  1. the happy path validating ALLOW with reason codes;
  2. the four canonical faults (F1-F4) being rejected with their codes;
  3. the v0.2-preview families (F5-F8) being rejected/quarantined;
  4. the guarded update refusing a non-ALLOW handle (fail-closed).

Run:  uv run python examples/countdown/demo.py
"""

from __future__ import annotations

import sys

from grpo_guard import testing
from grpo_guard.adapters.guarded_update import GuardedUpdateAdapter
from grpo_guard.faults import (
    inject_f1_static_rollout,
    inject_f2_misbound_logprob,
    inject_f3_retokenization,
    inject_f4_mask_shift,
)
from grpo_guard.faults.f5_f8 import (
    inject_f5_split_leakage,
    inject_f6_evaluator_alias,
    inject_f7_event_reorder,
    inject_f8_artifact_mutation,
)
from grpo_guard.schema.decisions import ValidationDecision
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope

STRICT = ProtocolConfig(name="strict_v01", mode="strict_on_policy")


def decide(t, stage="identity_pre_reward", update_input=None, **kw) -> str:
    ctx = ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=STRICT,
        split_registry=getattr(t, "split_registry", None),
        eval_protocol_sha256=getattr(t, "eval_protocol_sha256", None),
        update_input_event=update_input,
        **kw,
    )
    d = validate_envelope(ctx, stage)
    payload = d.decision_payload
    if payload.decision == "allow":
        return f"{payload.decision} [all {len(payload.reason_codes)} rules checked]"
    return f"{payload.decision} [{', '.join(payload.reason_codes[:3])}]"


def update_input_for(t) -> "UpdateInputEvent":
    """Minimal update input for this trajectory (F7 ordering check)."""
    from grpo_guard.schema.artifacts import EventRef
    from grpo_guard.schema.events import UpdateInputEvent

    gen = t.events[t.envelope.generation_event.event_id]
    return UpdateInputEvent(
        event_id=f"uinput-{t.envelope.envelope_id}", run_id=t.run_id,
        component_id="materializer", lifecycle_seq=gen.lifecycle_seq + 100,
        created_at_utc=gen.created_at_utc, update_id="update-1",
        preupdate_envelope=t.envelope.ref(),
        preupdate_validation_decision=EventRef(uri="", event_id="vdec-x", event_sha256="0" * 64),
        sequence_token_ids=gen.sequence_token_ids, loss_mask=gen.loss_mask,
        authoritative_behavior_logprob_event=t.envelope.training_contract.authoritative_behavior_logprob_event,
        authoritative_behavior_logprobs=gen.service_behavior_logprobs,
        reward_event=EventRef(uri="", event_id="reward-x", event_sha256="0" * 64),
        materialized_layout_sha256="0" * 64, single_use_nonce_sha256="0" * 64,
        tokenizer_called=False,
    ).seal()


def main() -> int:
    print("=" * 62)
    print("GRPO-Guard demo — trajectory contract & fault injection")
    print("=" * 62)

    print("\n[1] Happy path (normal trajectory)")
    t = testing.build_trajectory()
    print("    decision:", decide(t))

    print("\n[2] Canonical faults (v0.1 matrix, F1-F4, fresh base per case)")
    for name, make in [
        ("F1 static rollout     ", lambda: inject_f1_static_rollout(testing.build_trajectory(), 0, 1)),
        ("F2 misbound old-logp  ", lambda: inject_f2_misbound_logprob(testing.build_trajectory(), 1)),
        ("F3 retokenization     ", lambda: inject_f3_retokenization(testing.build_trajectory())),
        ("F4 mask shift (1 tok) ", lambda: inject_f4_mask_shift(testing.build_trajectory(), 1)),
    ]:
        print(f"    {name}:", decide(make()))

    print("\n[3] v0.2-preview families (F5-F8, full_pre_update stage)")
    print("    F5 split leakage   :",
          decide(inject_f5_split_leakage(testing.build_trajectory()), stage="full_pre_update"))
    f7 = inject_f7_event_reorder(testing.build_trajectory())
    print("    F7 event reorder   :",
          decide(f7, stage="full_pre_update", update_input=update_input_for(f7)))
    with testing.ArtifactStoreTmp() as store:
        t8 = testing.build_trajectory(store)
        print("    F8 artifact mutate :", decide(inject_f8_artifact_mutation(t8)))

    print("\n[4] Guarded update — fail closed without an ALLOW decision")
    t6 = testing.build_trajectory(stage="pre_update")
    from grpo_guard.schema.decisions import ValidationDecision

    parent = testing.validation_event(
        t6.run_id,
        ValidationDecision(decision="allow", validation_stage="identity_pre_reward", reason_codes=["G001"]),
        t6.next_seq(),
    )
    t6.events[parent.event_id] = parent
    from grpo_guard.schema.artifacts import EventRef

    t6.envelope = t6.envelope.model_copy(deep=True).model_copy(update={
        "envelope_id": f"{t6.envelope.envelope_id}-parented",
        "parent_identity_decision": EventRef(uri="", event_id=parent.event_id, event_sha256=parent.event_sha256),
    })
    t6.envelope.envelope_sha256 = ""
    t6.envelope = t6.envelope.seal()
    f6 = inject_f6_evaluator_alias(t6, "eval-proto-abc")
    print("    F6 evaluator alias :", decide(f6, stage="full_pre_update"))
    adapter = GuardedUpdateAdapter(t6.store, decision_verifier=lambda ref: False)
    try:
        adapter.update("prompt+completion text")
    except TypeError as exc:
        print("    text input refused :", type(exc).__name__)

    print("\n" + "=" * 62)
    print("All decisions are reason-coded and machine-readable (design doc §8).")
    print("Full evidence chain: artifacts/v0.1.0/ + artifacts/v0.2.0-dev/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
