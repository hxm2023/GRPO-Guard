"""Chaos/fuzz properties: validator never crashes and is deterministic on
arbitrary trajectory mutations (design doc §15.2, chaos engineering)."""

from __future__ import annotations

import numpy as np
import hypothesis.strategies as st
from hypothesis import given, settings

from grpo_guard import testing
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope

PROTOCOL = ProtocolConfig(name="strict_v01", mode="strict_on_policy")


def _ctx(t: testing.Trajectory) -> ValidationContext:
    return ValidationContext(
        envelope=t.envelope, store=t.store, events=t.events,
        policy_manifest=t.policy_manifest, split_manifest=t.split_manifest,
        protocol=PROTOCOL,
    )


@st.composite
def mutation(draw):
    """Pick a random single-field mutation of a valid trajectory."""
    kind = draw(st.sampled_from([
        "flip_token", "shift_mask", "corrupt_logprob", "shift_span",
        "change_prompt_id", "change_version", "truncate_seq", "noop",
    ]))
    return kind


@settings(max_examples=120)
@given(mutation())
def test_fuzz_never_crashes_and_deterministic(kind: str):
    t = testing.build_trajectory()
    gen = t.events[t.envelope.generation_event.event_id]
    if kind == "flip_token":
        t.sequence[5] = (t.sequence[5] + 1) % 32000
    elif kind == "shift_mask":
        t.target_mask = np.roll(t.target_mask, 1)
        t.loss_mask = t.target_mask[1:]
    elif kind == "corrupt_logprob":
        t.logprobs = np.full_like(t.logprobs, -13.37)
    elif kind == "shift_span":
        gen.completion_span = [gen.completion_span[0] + 1, gen.completion_span[1] + 1]
    elif kind == "change_prompt_id":
        gen.prompt_id = "fuzz-prompt"
    elif kind == "change_version":
        gen.behavior_policy_version = 7
    elif kind == "truncate_seq":
        t.sequence = t.sequence[:-1]
        t.loss_mask = t.loss_mask[:-1]
    # noop: exercise the clean path

    d1 = validate_envelope(_ctx(t), "identity_pre_reward").decision_payload
    d2 = validate_envelope(_ctx(t), "identity_pre_reward").decision_payload
    # deterministic: same input -> same decision and codes
    assert d1.decision == d2.decision
    assert d1.reason_codes == d2.reason_codes
    # decision is always a valid one (never crashes, never ambiguous)
    assert d1.decision in ("allow", "quarantine", "reject")


def test_clean_trajectory_allows():
    t = testing.build_trajectory()
    d = validate_envelope(_ctx(t), "identity_pre_reward").decision_payload
    assert d.decision == "allow"


def test_validator_latency_regression():
    """Guard overhead must not regress: 64 validations well under 50 ms/env
    (measured 0.6-1.0 ms/env on CI-class CPU)."""
    import time

    t = testing.build_trajectory()
    ctx = _ctx(t)
    n = 64
    t0 = time.perf_counter()
    for _ in range(n):
        validate_envelope(ctx, "identity_pre_reward")
    elapsed = time.perf_counter() - t0
    per_env = elapsed / n * 1000.0
    assert per_env < 50.0, f"validator latency regressed: {per_env:.1f} ms/env"
