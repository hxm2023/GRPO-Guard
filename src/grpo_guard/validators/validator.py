"""Validator orchestrator: run the staged rule set, aggregate, produce a
sealed ValidationDecision (design doc §8, §7.8)."""

from __future__ import annotations

from grpo_guard.schema.decisions import (
    PRE_UPDATE_STAGE_RULES,
    Decision,
    ValidationDecision,
    ValidationStage,
)
from grpo_guard.schema.events import ValidationDecisionEvent
from grpo_guard.validators.context import ValidationContext
from grpo_guard.validators.rules import ALL_RULES, ruleset_sha256

def _expand(short_codes: frozenset[str]) -> frozenset[str]:
    """Map stage membership (short codes) to full rule names in ALL_RULES."""
    return frozenset(code for code in ALL_RULES if code.split("_", 1)[0] in short_codes)


RULES_BY_STAGE: dict[ValidationStage, frozenset[str]] = {
    "identity_pre_reward": _expand(frozenset(
        c for c in PRE_UPDATE_STAGE_RULES
        if c.startswith(("P", "T", "M", "L")) or c == "R004"
    )),
    "full_pre_update": _expand(PRE_UPDATE_STAGE_RULES),
}


def validate_envelope(
    ctx: ValidationContext,
    stage: ValidationStage,
    validator_version: str = "1.0",
) -> ValidationDecisionEvent:
    """Run every rule required for ``stage``; aggregate; seal the decision.

    - any reject → reject
    - else any quarantine → quarantine
    - else allow (all required rules ran without violation)
    """
    fired_reject: list[str] = []
    fired_quarantine: list[str] = []
    checked: list[str] = []
    for code in sorted(RULES_BY_STAGE[stage]):
        fn = ALL_RULES[code]
        result = fn(ctx)
        checked.append(code)
        if result is None:
            continue
        if result.decision == "reject":
            fired_reject.append(result.code)
        elif result.decision == "quarantine":
            fired_quarantine.append(result.code)
        ctx.notes.append(f"{result.code}: {result.decision} — {result.detail}")

    if fired_reject:
        decision: Decision = "reject"
        reason_codes = fired_reject
    elif fired_quarantine:
        decision = "quarantine"
        reason_codes = fired_quarantine
    else:
        decision = "allow"
        reason_codes = checked

    checked_events = {
        ctx.envelope.generation_event.event_id,
        ctx.envelope.scoring_event.event_id if ctx.envelope.scoring_event else None,
        ctx.envelope.reward_event.event_id if ctx.envelope.reward_event else None,
        ctx.envelope.parent_identity_decision.event_id if ctx.envelope.parent_identity_decision else None,
    } - {None}
    checked_artifacts: set[str] = set()
    for ev in ctx.events.values():
        for ref in list(ev.input_artifacts) + list(ev.output_artifacts):
            checked_artifacts.add(ref.sha256)

    payload = ValidationDecision(
        decision=decision,
        validation_stage=stage,
        reason_codes=reason_codes,
        checked_ruleset_sha256=ruleset_sha256(),
        checked_event_sha256s=sorted(
            {ctx.events[eid].event_sha256 for eid in checked_events if eid in ctx.events}
        ),
        checked_artifact_sha256s=sorted(checked_artifacts),
        observed_policy_lag=ctx.protocol.max_policy_lag_versions,
        validator_version=validator_version,
    )
    # observed lag computed from the generation event when available
    gen = ctx.event(ctx.envelope.generation_event.event_id)
    if gen is not None and hasattr(gen, "behavior_policy_version"):
        payload.observed_policy_lag = max(
            0, ctx.envelope.training_contract.trainer_parent_policy_version - gen.behavior_policy_version
        )

    event = ValidationDecisionEvent(
        event_id=f"vdec-{ctx.envelope.envelope_id}",
        event_type="validation_decision",
        run_id=ctx.envelope.run_id,
        component_id="validator",
        lifecycle_seq=_next_seq(ctx),
        created_at_utc=_now_utc(),
        input_events=[ctx.envelope.generation_event],
        decision_payload=payload,
    ).seal()
    return event


def _next_seq(ctx: ValidationContext) -> int:
    return max([e.lifecycle_seq for e in ctx.events.values()] + [-1]) + 1


def _now_utc() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
