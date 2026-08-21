"""Fault injector contract: each canonical fault produces its reason code
(design doc §11, §12.3)."""

from grpo_guard import testing
from grpo_guard.faults import (
    inject_f1_static_rollout,
    inject_f2_misbound_logprob,
    inject_f3_retokenization,
    inject_f4_mask_shift,
)
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope

STRICT = ProtocolConfig(name="strict_v01", mode="strict_on_policy")


def decide(t, stage="identity_pre_reward"):
    ctx = ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=STRICT,
    )
    return validate_envelope(ctx, stage).decision_payload


def test_f1_static_rollout_rejected_strict():
    t = testing.build_trajectory()
    f = inject_f1_static_rollout(t, runtime_version=0, claimed_parent=1)
    d = decide(f)
    assert d.decision == "reject"
    assert "P004_STALE_POLICY_STRICT" in d.reason_codes
    assert d.observed_policy_lag == 1


def test_f1_normal_neighbor_passes():
    t = testing.build_trajectory()
    d = decide(t)
    assert d.decision == "allow"


def test_f2_misbound_logprob_rejected():
    t = testing.build_trajectory()
    f = inject_f2_misbound_logprob(t, scorer_policy_version=1)
    d = decide(f)
    assert d.decision == "reject"
    assert "L003_SCORER_POLICY_MISMATCH" in d.reason_codes


def test_f3_retokenization_rejected():
    t = testing.build_trajectory()
    f = inject_f3_retokenization(t)
    d = decide(f)
    assert d.decision == "reject"
    assert "T002_TOKENIZER_MISMATCH" in d.reason_codes


def test_f4_mask_shift_rejected():
    t = testing.build_trajectory()
    f = inject_f4_mask_shift(t, shift=1)
    d = decide(f)
    assert d.decision == "reject"
    assert "M004_CANONICAL_MASK_MISMATCH" in d.reason_codes


def test_f4_shift_boundary_3_tokens():
    t = testing.build_trajectory()
    f = inject_f4_mask_shift(t, shift=3)
    d = decide(f)
    assert d.decision == "reject"


def test_faults_never_alter_artifacts_other_than_target():
    t = testing.build_trajectory()
    f = inject_f4_mask_shift(t, shift=1)
    assert f.sequence.tobytes() == t.sequence.tobytes()
    assert f.logprobs.tobytes() == t.logprobs.tobytes()
