"""Weight-identity canary suite (design doc §10).

Two-layer evidence: control-plane (manifest/version/load-epoch) plus
data-plane greedy token sketches over a fixed prompt suite.  The suite is
calibrated by reloading the SAME checkpoint ≥5 times to measure natural
drift, then the tolerance is frozen.  A canary only attests behavior
consistency in the pinned environment — it is not a per-byte proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CANARY_PROMPTS: list[str] = [
    "2+2=",  # short text/arithmetic
    "Multiply 3, 7 and 11: ",  # numbers
    "List the special token ids of this tokenizer: ",  # special tokens
    "Complete this long sentence with exactly five words: " + "word " * 900,  # near max-context
]


@dataclass
class CanaryResult:
    policy_version: int
    sketches: list[list[int]] = field(default_factory=list)  # per-prompt token ids
    drift: dict = field(default_factory=dict)
    verdict: str = "unknown"  # pass | mismatch


class CanarySuite:
    def __init__(self, prompts: list[str] | None = None, max_tokens: int = 8):
        self.prompts = prompts or CANARY_PROMPTS
        self.max_tokens = max_tokens

    def sketch(self, generate_fn) -> list[list[int]]:
        """Greedy token sketch per prompt.  ``generate_fn`` must return
        (prompt_ids, completion_ids, logprobs, logprob_token_ids)."""
        sketches = []
        for prompt in self.prompts:
            res = generate_fn([prompt], n=1, temperature=0.0, top_p=1.0, max_tokens=self.max_tokens)
            if isinstance(res, dict):
                # TRL's VLLMClient.generate returns a dict keyed by
                # prompt_ids/completion_ids/logprobs/logprob_token_ids;
                # tuple-unpacking a dict yields its KEYS — a silent constant
                # sketch.  This bug made every canary look identical
                # (drift always 0) and is fixed here.
                completion_ids = res["completion_ids"][0]
            else:
                _, completion_ids, _, _ = res
                completion_ids = completion_ids[0]
            sketches.append(completion_ids)
        return sketches

    def calibrate(self, generate_fn, reloads: int = 5) -> dict:
        """Measure natural drift across repeated loads of the same weights."""
        all_sketches = [self.sketch(generate_fn) for _ in range(reloads)]
        baseline = all_sketches[0]
        drift = 0
        for other in all_sketches[1:]:
            for b, o in zip(baseline, other):
                drift = max(drift, _token_diff(b, o))
        return {"reloads": reloads, "observed_max_token_drift": drift, "frozen_tolerance": max(drift, 0)}

    def check(self, generate_fn, policy_version: int, baseline: list[list[int]], tolerance: int = 0) -> CanaryResult:
        sketches = self.sketch(generate_fn)
        drift = max(_token_diff(b, s) for b, s in zip(baseline, sketches))
        verdict = "pass" if drift <= tolerance else "mismatch"
        return CanaryResult(policy_version=policy_version, sketches=sketches, drift={"max_token_drift": drift}, verdict=verdict)


def _token_diff(a: list[int], b: list[int]) -> int:
    n = max(len(a), len(b))
    if n == 0:
        return 0
    return sum(1 for i in range(n) if i >= len(a) or i >= len(b) or a[i] != b[i])
