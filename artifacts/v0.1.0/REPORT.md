# GRPO-Guard v0.1.0 — run report

Created: 2026-08-22. Commit-bound evidence: every number below traces to
`artifacts/v0.1.0/*` + git commit + SHA256SUMS (design doc §16, §19.2).

## Day 1 — Compatibility Gate (PASSED)

- Profile frozen at `compatibility_profile.yaml`: python 3.12.3, torch
  2.11.0+cu130, trl 1.10.0, vllm 0.26.0, transformers 5.15.1, Qwen3-4B @
  1cfa9a7208912126459214e8b04321603b3df60c, autodl2 (2xRTX6000D).
- Official TRL+vLLM server-mode smoke green: `trl vllm-serve` (GPU1) +
  GRPOTrainer `vllm_mode="server"` (GPU0), one committed optimizer step,
  **398 observed `update_named_param` weight-sync calls** (policy v0→v1).
  Evidence: `artifacts/v0.1.0/smoke/`.
- Adapter patches (design doc §14.2): device-index normalization for
  trl 1.10.0/vllm 0.26.0 pynccl; torchcodec removed (broken .so).
  See DECISION_LOG.md D1/D2.
- CPU contract suite: 95+ unit/contract/property tests green
  (`uv run pytest tests/`).

## Day 2 — Real guarded online closed loop (PASSED)

Evidence: `artifacts/v0.1.0/loop/` (run_manifest.json, REPORT.md, full
append-only event lineage + content-addressed artifacts, 1.2MB; checkpoints
not published per design doc §18).

| stage | result |
|---|---|
| canary calibration | 5 reloads of the same checkpoint, frozen tolerance = 0 |
| v0 rollout | 32 sequences (8 prompts × 4 gens), identity validation **ALLOW on all 32** |
| pre-update validation | **ALLOW on all 32** envelopes; handles materialized |
| guarded update | 1 real optimizer step, B=32, ratio p50=1.000 / p95=1.083 / max=2.434, clip 2.2%, loss=0.0000 (behavior==policy ⇒ ratio≈1; gradient-impact evidence is Day 4) |
| v1 commit | content-hashed PolicyManifest (`a61b4009…`), v0 `13712047…` |
| sync | 398 observed `update_named_param` calls |
| canary v1 | **pass** (max token drift 0) |
| v1 rollout | validated (identity ALLOW) |

## Day 3 — F1-F4 fault matrix + Correctness Gate (PASSED)

Evidence: `artifacts/v0.1.0/fault_matrix.json` (computed from the REAL Day 2
loop events, not synthetic fixtures).

- canonical F1–F4: **4/4 reject** with pre-registered reason codes
  (P004 / L003 / T002 / M002+M004)
- all 12 fault variants (canonical + boundary + held-out): 12/12 match
  pre-registered expectations; held-out variants exercise distinct rule
  branches (P003 stale-sync, L006 wrong-generation, T004 re-encoded
  sequence, M004 right-truncation shift)
- normal set: **32/32 ALLOW**, 0 quarantine, 0 reject (false-reject = 0)
- boundary cases: 4/4 (empty completion → M005 quarantine; left padding /
  truncation / dual-source-diagnostic → allow)
- strict stale-trajectory acceptance: **0** (computed from results)
- quarantine/reject never enters consumption (guarded update requires an
  ALLOW decision verifier + single-use nonce registry)

Cross-model review (reviewer-backend: claude-subagent; codex-mcp 403 quota):
REVISE round found shared-base contamination, missing ALLOW verification,
hardcoded stale-acceptance, overclaimed v1 count — all fixed and re-run
green. See git history for the fix commits.

## Limitations (honest, design doc §21)

- loss = 0.0 on the Day 2 update because behavior policy == new policy
  (ratio ≈ 1); gradient-impact numbers come from Day 4 paired replay.
- canary is a behavior sketch (greedy tokens), not a per-byte proof
  (design doc §5.3, §10).
- v1 rollout count reflects what the server returned per request
  (`run_manifest.json`).

## Day 4 — Paired replay + overhead + Impact/Overhead Gate (PASSED)

Evidence: `artifacts/v0.1.0/replay/gradient_replay.json`, `overhead.json`.

**Paired gradient replay** (real 4B model, v1 weights + deterministic drift
seed=7 σ=0.02, real prompt group of 4 v0 trajectories with real rewards):

| pair | gradient cosine | relative L2 | control norm | fault norm |
|---|---|---|---|---|
| F2 misbound logprobs | 0.999 | 0.044 | 2.23e-2 | 2.24e-2 |
| F3 re-encoded tokens | **−0.094** | **139.0** | 2.23e-2 | **3.10** |
| F4 mask shift | **−0.031** | **1.00** | 2.23e-2 | **3.7e-4** |

- F3 (retokenization): the update direction flips and the gradient norm
  grows 140× — silent consumption would change the update completely.
- F4 (mask shift): the shifted mask nearly zeroes the gradient (norm 60×
  smaller) — the update silently loses its signal.
- F2 (misbound logprobs with value-close values): gradient barely moves —
  the contract (L003/L007) catches what values cannot; the replay shows
  the limit of value-level detection honestly.

**F1 guard-off accident**: `||θ_v1 − θ_v0|| = 0.0` (computed from the
committed checkpoints) — the Day 2 update had zero gradient (behavior==
policy ⇒ ratio≈1), so the stale-trajectory accident would also produce a
0.0 update norm. Reported as measured; no fabricated cosine.

**Counterexample (loss looks normal, contract fails)**: control loss
6.9e-7 vs F4-fault loss 5.5e-9 and F2-fault 6.9e-7 — loss/ratio metrics are
indistinguishable, while the contract rejects the faulted envelopes with
M004/L003. This is the exact legacy pattern (trainer loss/KL looked fine
under a static rollout).

**Overhead** (fixed workload: 8 real envelopes, 3 repeats, raw + mean ±
stdev): guard-on 51.6 ± 0.2 ms vs guard-off 2.8 ± 0.3 ms per batch →
48.9 ms/batch (~6.1 ms per envelope), all three repeats reported in
`overhead.json`.

**Stage timings** (from event timestamps): sync 53 s, rollout 52 s,
validation 16 s, reward 16 s, update+commit ~40 s (Day 2 loop).
