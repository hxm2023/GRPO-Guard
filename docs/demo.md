# GRPO-Guard demo — 3-5 minute walkthrough

Run: `uv run python examples/countdown/demo.py` (CPU only, no GPU needed).

## Script

**1. Happy path (30 s)** — a canonical trajectory validates `allow` with all
27 rules checked.  Point out: the decision is a sealed, reason-coded event;
nothing was mocked.

**2. The four canonical faults F1–F4 (90 s)** — each injector mutates exactly
one field of a fresh trajectory:

| Fault | Injection | Decision |
|---|---|---|
| F1 static rollout | generation claims runtime v0, contract claims parent v1 | `reject P004_STALE_POLICY_STRICT` |
| F2 misbound old-logprob | v1 scorer bound to a v0 generation | `reject L003_SCORER_POLICY_MISMATCH` |
| F3 retokenization | tokenizer identity differs from the manifest | `reject T002_TOKENIZER_MISMATCH` |
| F4 mask shift | completion mask moved by 1 token | `reject M002_PROMPT_SELECTED + M004_CANONICAL_MASK_MISMATCH` |

Key line for interviews: *"the loss/KL looked normal in the legacy incident —
the contract catches what the curves cannot."*

**3. v0.2-preview families F5–F8 (60 s)** — split leakage (D003), evaluator
alias (R006 → quarantine), event reorder (L005), artifact mutation (T001).
Clearly scoped as v0.2-preview per the design doc.

**4. Fail-closed guarded update (30 s)** — the update adapter refuses text
input with a `TypeError`; the F6 quarantine proves non-ALLOW envelopes never
reach the optimizer.

## Talking points

- Producer ownership: the runtime adapter is the ONLY producer of token
  artifacts; the trainer never re-tokenizes.
- The evidence chain: every number in README/REPORT traces to
  `artifacts/v0.1.0/` + commit + SHA256SUMS (design doc §16).
- Honesty: the v0.1 update had loss ≈ 0 (behavior == policy) — gradient
  impact comes from the Day 4 paired replay; the canary is a behavior
  sketch, not a byte proof.
