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
  (design doc §6.3 lineage gap, found in Day 4 review); fixed and re-run —
  the committed `artifacts/v0.1.0/loop/` evidence is the FINAL loop run and
  includes `upd-commit-update-1`.
- Paired replay runs at v1 weights + a deterministic drift (seed 7,
  σ=0.005) because the v0.1 single update moved weights ~0 — the replay
  state is a documented simulation of a later training state, not a claim
  about v0/v1 distance (day4_summary.json).

## Day 4 — Paired replay + overhead + Impact/Overhead Gate (PASSED)

Evidence: `artifacts/v0.1.0/replay/gradient_replay.json`, `overhead.json`.

**Paired gradient replay** (real 4B model, v1 weights + deterministic drift
seed=7 σ=0.005 — the smallest bf16-meaningful drift that keeps a sane
on-policy loss regime; real prompt group of 4 v0 trajectories with real
rewards [0,0,0,1]; one value flipped in degenerate all-equal groups,
disclosed in day4_summary.json).  The replay ran against the FIRST loop
run (loop-1787355683); its inputs are preserved in git history (commit
31c3c39), while the published `loop/` is the FINAL run (loop-1787364111)
with identical gate results.

| pair | gradient cosine | relative L2 | control norm | fault norm | loss_c | loss_f |
|---|---|---|---|---|---|---|
| F2 misbound logprobs | 0.989 | 0.148 | 6.92 | 6.96 | −0.0204 | −0.0193 |
| F3 re-encoded tokens | **0.634** | **0.794** | 6.92 | **5.63** | −0.0204 | −0.0076 |
| F4 mask shift | **0.238** | **4.69** | 6.92 | **33.4** | −0.0204 | −0.0336 |

- F4 (mask shift): cosine drops to 0.24 and the gradient norm grows ~5× —
  the shifted mask changes both the update direction and scale.
- F3 (retokenization): cosine 0.63 — the re-encoded sequence materially
  shifts the update direction (loss changes 2.7×).
- F2 (misbound logprobs with value-close values): cosine 0.99 — the
  gradient barely moves; the contract (L003/L007) catches what values
  cannot; the replay shows the limit of value-level detection honestly.
- selected_prompt_tokens / selected_padding_tokens (§12.4): 0 / 0 for the
  control mask (recorded in gradient_replay.json).

**F1 guard-off accident**: `||θ_v1 − θ_v0|| = 0.0` measured from the
committed checkpoint shards at fp32 precision (method + precision caveat in
`day4_summary.json`). The Day 2 update's gradient was ~1e-6-scale (bf16
rounding made ratio ≠ 1 on ~2.2% of tokens, cf. manifest ratio_max 2.434 /
clip 2.2%), which is below fp32 weight resolution — the committed weights
are identical at fp32. The stale-trajectory accident consumes the same
tensors, so its update norm is the same 0.0; reported as measured, no
fabricated cosine.

**Counterexample (loss looks normal, contract fails)**: control loss
−0.0204 vs F2-fault loss −0.0193 (5% apart, both "normal-looking" negative
GRPO losses) — loss-level metrics are indistinguishable while the contract
rejects the misbound envelope with L003/L007. This is the exact legacy
pattern (trainer loss/KL looked fine under a static rollout).

**Overhead** (fixed workload: 8 real envelopes from the FINAL loop evidence,
3 repeats, raw + mean ± stdev): guard-on 42.8 ± 4.2 ms vs guard-off
2.36 ± 0.34 ms per batch → 40.4 ms/batch (~5.1 ms per envelope); all raw
values in `overhead.json`.

**Stage timings** (`stage_timings.json`, from event timestamps of the FINAL
loop): sync 53.4 s, rollout 53.1 s, validation 16.4 s, reward 16.1 s,
update_committed event present.  (The first loop run lacked the
`update_committed` event — gap found in review, fixed; the published
`loop/` is the rerun.)
