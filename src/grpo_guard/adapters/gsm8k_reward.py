"""Deterministic GSM8K-style rule verifier (framework portability demo).

A second task adapter showing the framework is NOT bound to Countdown:
same TrajectoryEnvelope / RewardEvent / validator pipeline, different
deterministic rule set.  The verifier extracts the LAST numeric value in
the completion and compares it against the golden answer (float
tolerance).  Deterministic and versioned — any rule change must bump
``reward_version`` and the protocol hash or rewards lose lineage identity.
"""

from __future__ import annotations

import hashlib
import re

from grpo_guard.store.canonical_json import canonical_dumps

REWARD_VERSION = "gsm8k-rule-v1"

_ANSWER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_TOLERANCE = 1e-6


def gsm8k_rule_verifier(completion: str, golden_answer: float) -> dict:
    """Return reward components for one completion.

    correctness: 1.0 iff the last numeric value in the completion equals
    the golden answer within tolerance; format: 1.0 iff a numeric value
    was extractable (completion non-empty, answer-shaped).
    """
    components = {"correctness": 0.0, "format": 0.0}
    matches = _ANSWER_RE.findall(completion or "")
    if not matches:
        return components
    components["format"] = 1.0
    try:
        last = float(matches[-1])
    except ValueError:
        return components
    if abs(last - float(golden_answer)) <= _TOLERANCE:
        components["correctness"] = 1.0
    return components


def reward_protocol_sha256() -> str:
    """Identity of the verifier rule set; bound into every RewardEvent."""
    return hashlib.sha256(
        canonical_dumps(
            {
                "reward_version": REWARD_VERSION,
                "extraction_rule": "last-numeric-value",
                "format_rule": "numeric-extractable",
                "float_tolerance": _TOLERANCE,
            }
        )
    ).hexdigest()


class Gsm8kRewardAdapter:
    """Produce reward components with explicit protocol identity (§7.6)."""

    def __init__(self, protocol_sha256: str | None = None):
        self.protocol_sha256 = protocol_sha256 or reward_protocol_sha256()

    def score(self, completion: str, golden_answer: float) -> dict:
        return gsm8k_rule_verifier(completion, golden_answer)
