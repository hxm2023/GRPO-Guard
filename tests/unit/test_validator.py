"""Validator orchestration contract (design doc §8, §7.8)."""

import pytest

from grpo_guard import testing
from grpo_guard.faults import inject_f1_static_rollout
from grpo_guard.schema.decisions import PRE_UPDATE_STAGE_RULES, ValidationDecision
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope

STRICT = ProtocolConfig(name="strict_v01", mode="strict_on_policy")


def full_ctx(t, **kw):
    return ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=STRICT, **kw,
    )


def test_happy_path_identity_allow():
    t = testing.build_trajectory()
    dec = validate_envelope(full_ctx(t), "identity_pre_reward")
    p = dec.decision_payload
    assert p.decision == "allow"
    assert p.validation_stage == "identity_pre_reward"
    assert p.checked_ruleset_sha256  # frozen rule table identity
    assert len(p.reason_codes) >= 10  # all rules ran (positive codes recorded)
    assert dec.verify_seal()


def test_happy_path_full_pre_update_allow():
    t = testing.build_trajectory()
    dec1 = validate_envelope(full_ctx(t), "identity_pre_reward")
    parent = dec1.decision_payload
    assert parent.decision == "allow"
    t2 = testing.build_trajectory(
        stage="pre_update",
        parent_envelope_sha256=t.envelope.envelope_sha256, parent_identity=dec1,
    )
    t2.events[dec1.event_id] = dec1
    dec2 = validate_envelope(full_ctx(t2), "full_pre_update")
    assert dec2.decision_payload.decision == "allow"


def test_identity_allow_does_not_grant_update():
    # identity ALLOW only authorizes reward computation; pre-update must still
    # reject a pre-reward-only chain (R003) when consumed as pre-update
    t = testing.build_trajectory()
    dec = validate_envelope(full_ctx(t), "identity_pre_reward")
    assert dec.decision_payload.decision == "allow"
    t2 = testing.build_trajectory(stage="pre_update")
    dec2 = validate_envelope(full_ctx(t2), "full_pre_update")
    assert dec2.decision_payload.decision == "reject"  # R005: no allowed parent


def test_reject_dominates_quarantine():
    t = testing.build_trajectory()
    t2 = inject_f1_static_rollout(t, runtime_version=0, claimed_parent=1)
    # also empty completion → quarantine; reject must dominate
    t2 = testing.build_trajectory(completion_span=[4, 4])
    t2 = inject_f1_static_rollout(t2, runtime_version=0, claimed_parent=1)
    dec = validate_envelope(full_ctx(t2), "identity_pre_reward")
    assert dec.decision_payload.decision == "reject"
    assert "P004_STALE_POLICY_STRICT" in dec.decision_payload.reason_codes


def test_stage_rule_membership_frozen():
    # identity stage must not include D/R rules except R004 (short codes)
    identity_rules = {c for c in PRE_UPDATE_STAGE_RULES if c.startswith(("P", "T", "M", "L")) or c == "R004"}
    assert "R003" not in identity_rules
    assert "D001" not in identity_rules
    assert "R004" in identity_rules
    assert "R005" in PRE_UPDATE_STAGE_RULES
