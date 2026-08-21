"""Deterministic paired-replay derivation (design doc §12.2-§12.3).

Pairs freeze the SAME base artifacts (checkpoint identity, tokens, masks,
rewards, RNG) and differ in exactly one target field:
  F2: token/mask unchanged, only the scoring event + logprob ref replaced;
  F3: same text, different token artifact (deterministic re-encode);
  F4: only the mask moved; token/logprob untouched.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from grpo_guard import testing
from grpo_guard.faults import (
    inject_f2_misbound_logprob,
    inject_f3_retokenization,
    inject_f4_mask_shift,
)


@dataclass
class ReplayPair:
    fault_id: str
    control_envelope: dict
    fault_envelope: dict
    base_artifacts: dict
    base_sha256: str
    frozen_at: Path | None = None


def freeze_base_artifacts(t: testing.Trajectory) -> dict:
    """Serialize the replay base: tokens, masks, logprobs, rewards, layout."""
    gen = t.events[t.envelope.generation_event.event_id]
    return {
        "sequence_token_ids": t.sequence.tolist(),
        "completion_target_mask": t.target_mask.tolist(),
        "loss_mask": t.loss_mask.tolist(),
        "behavior_logprobs": t.logprobs.tolist(),
        "rewards": {"correctness": t.reward_components.get("correctness", 0.0),
                    "format": t.reward_components.get("format", 0.0)},
        "policy_version": gen.behavior_policy_version,
        "completion_span": gen.completion_span,
    }


def derive_fault_pair(t: testing.Trajectory, fault_id: str, variant: dict) -> ReplayPair:
    """Derive the (control, fault) envelope pair from one base trajectory."""
    base = freeze_base_artifacts(t)
    base_sha = hashlib.sha256(repr(sorted(base.items())).encode()).hexdigest()

    if fault_id == "f2_misbound_logprob":
        fault = inject_f2_misbound_logprob(t, variant["scorer_policy_version"])
    elif fault_id == "f3_retokenization":
        fault = inject_f3_retokenization(t)
    elif fault_id == "f4_mask_shift":
        fault = inject_f4_mask_shift(t, variant["shift"])
    else:
        raise ValueError(fault_id)

    return ReplayPair(
        fault_id=fault_id,
        control_envelope={"envelope_id": t.envelope.envelope_id, "envelope_sha256": t.envelope.envelope_sha256},
        fault_envelope={"envelope_id": fault.envelope.envelope_id, "envelope_sha256": fault.envelope.envelope_sha256},
        base_artifacts=base,
        base_sha256=base_sha,
    )
