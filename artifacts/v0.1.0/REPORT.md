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
- The Day 2 loop's first run did not emit the `update_committed` event
  (design doc §6.3 lineage gap, found in Day 4 review); fixed in the loop,
  final run scheduled for the Day 5 release evidence.
- Paired replay runs at v1 weights + a deterministic drift (seed 7,
  σ=0.005) because the v0.1 single update moved weights ~0 — the replay
  state is a documented simulation of a later training state, not a claim
  about v0/v1 distance (day4_summary.json).

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

**F1 guard-off accident**: `||θ_v1 − θ_v0|| = 0.0` measured from the
committed checkpoint shards at fp32 precision (method + precision caveat in
`day4_summary.json`). The Day 2 update's gradient was ~1e-6-scale (bf16
rounding made ratio ≠ 1 on ~2.2% of tokens, cf. manifest ratio_max 2.434 /
clip 2.2%), which is below fp32 weight resolution — the committed weights
are identical at fp32. The stale-trajectory accident consumes the same
tensors, so its update norm is the same 0.0; reported as measured, no
fabricated cosine.

**Counterexample (loss looks normal, contract fails)**: control loss
6.9e-7 vs F2-fault loss 6.9e-7 (1% apart, ratio p50 both ≈ 1) — loss/ratio
metrics are indistinguishable while the contract rejects the misbound
envelope with L003/L007. Same for F4: the masked loss (5.5e-9) is still a
"normal-looking" tiny number while M002/M004 fire. This is the exact legacy
pattern (trainer loss/KL looked fine under a static rollout).

**Overhead** (fixed workload: 8 real envelopes, 3 repeats, raw + mean ±
stdev): guard-on 51.6 ± 0.2 ms vs guard-off 2.75 ± 0.5 ms per batch →
48.9 ms/batch (~6.1 ms per envelope); all raw values in `overhead.json`.

**Stage timings** (`stage_timings.json`, from event timestamps): sync
events span 53 s, rollout events 52 s, validation 16 s, reward 16 s (Day 2
loop).  Note: the loop's first run did NOT emit an `update_committed` event
(gap found in review); the loop now emits it (rerun for the Day 5 release
evidence).
