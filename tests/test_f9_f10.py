"""F9-F10 injector + validator contract tests (v0.2.1, D16)."""

from __future__ import annotations

from grpo_guard import testing
from grpo_guard.faults.f9_f10 import inject_f9_reward_hacking, inject_f10_data_poisoning
from grpo_guard.schema.decisions import ValidationDecision
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope

PROTOCOL = ProtocolConfig(name="strict_v01", mode="strict_on_policy")


def _pre_update_trajectory() -> testing.Trajectory:
    parent = testing.validation_event(
        "run-f9f10", ValidationDecision(decision="allow", validation_stage="identity_pre_reward",
                                        reason_codes=["G001"]), 1)
    t = testing.build_trajectory(stage="pre_update", parent_identity=parent)
    t.events[parent.event_id] = parent  # R005 needs the parent event present
    return t


def ctx_for(t: testing.Trajectory) -> ValidationContext:
    return ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=PROTOCOL,
        reward_verifier_registry=getattr(t, "reward_verifier_registry", None),
    )


def test_f9_canonical_reject():
    t = inject_f9_reward_hacking(_pre_update_trajectory())
    d = validate_envelope(ctx_for(t), "full_pre_update").decision_payload
    assert d.decision == "reject"
    assert "R008_REWARD_VERIFIER_UNREGISTERED" in d.reason_codes


def test_f9_held_out_wrong_protocol_reject():
    t = inject_f9_reward_hacking(_pre_update_trajectory(),
                                 fake_version="countdown-rule-v1", wrong_protocol="f" * 64)
    d = validate_envelope(ctx_for(t), "full_pre_update").decision_payload
    assert d.decision == "reject"
    assert "R008_REWARD_VERIFIER_UNREGISTERED" in d.reason_codes


def test_f9_normal_allow():
    from grpo_guard.adapters.countdown_reward import reward_protocol_sha256

    t = _pre_update_trajectory()
    t.reward_verifier_registry = {"countdown-rule-v1": reward_protocol_sha256()}
    d = validate_envelope(ctx_for(t), "full_pre_update").decision_payload
    assert d.decision == "allow"


def _register_prompt_content(t: testing.Trajectory) -> testing.Trajectory:
    import hashlib

    import numpy as np

    gen = t.events[t.envelope.generation_event.event_id]
    start, end = gen.prompt_span
    seq = np.frombuffer(t.store.get(gen.sequence_token_ids), dtype=np.int32)
    t.split_manifest.content_sha256s[gen.prompt_id] = hashlib.sha256(seq[start:end].tobytes()).hexdigest()
    return t


def test_f10_canonical_reject():
    t = inject_f10_data_poisoning(_register_prompt_content(_pre_update_trajectory()))
    d = validate_envelope(ctx_for(t), "full_pre_update").decision_payload
    assert d.decision == "reject"
    assert "D004_PROMPT_CONTENT_MISMATCH" in d.reason_codes


def test_f10_normal_allow():
    t = _register_prompt_content(_pre_update_trajectory())
    d = validate_envelope(ctx_for(t), "full_pre_update").decision_payload
    assert d.decision == "allow"
