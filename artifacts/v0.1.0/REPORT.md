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

## v0.2 — F5-F8 fault families (formal, decision D12 2026-08-23)

Evidence: `artifacts/v0.2.0-dev/fault_matrix.json`, `tests/frozen/f5_f8_v02/`
(pre-registered expectations in `configs/faults/f5_f8_v02.yaml`).

| family | rule | decision |
|---|---|---|
| F5 split leakage | D003_SPLIT_OVERLAP | reject (overlap report) |
| F6 evaluator alias | R006_EVALUATOR_ALIAS | quarantine |
| F7 event reorder | L005_SCORING_AFTER_UPDATE | reject |
| F8 artifact mutation | T001_ARTIFACT_HASH_MISMATCH | reject |

v0.2 matrix over the REAL loop artifacts: **4/4 matched**, normal 4/4 allow.
Formal v0.2 families per decision D12 (design doc §11 upgrade condition
met: `docs/INJECTION_PROTOCOL_v02.md` frozen + all matrices pass) — NOT
part of the v0.1 matrix; no GPU budget spent.

## Online matrix + release reproducibility (2026-08-22, autodl2)

**F1-F8 online matrix** (`artifacts/v0.1.0/online/fault_matrix_online.json`,
`artifacts/v0.2.0-dev/fault_matrix_online.json`): all eight fault families
injected into GenerationEvents produced by a REAL vLLM server in the same
run (8 real rollouts); validator decides live:

| family | decision | reason codes |
|---|---|---|
| F1 static rollout | reject | P004_STALE_POLICY_STRICT |
| F2 misbound logprob | reject | L003_SCORER_POLICY_MISMATCH |
| F3 retokenization | reject | T002_TOKENIZER_MISMATCH |
| F4 mask shift | reject | M002_PROMPT_SELECTED + M004 |
| F5 split leakage | reject | D003_SPLIT_OVERLAP |
| F6 evaluator alias | quarantine | R006_EVALUATOR_ALIAS |
| F7 event reorder | reject | L005_SCORING_AFTER_UPDATE |
| F8 artifact mutation | reject | T001_ARTIFACT_HASH_MISMATCH |

normal set (real generations): 4/4 ALLOW.  F5-F8 are formal v0.2 families
per decision D12.

**Release reproducibility**: tag `v0.1.0` checked out and the official
server-mode smoke re-run on autodl2 from that exact commit →
`artifacts/v0.1.0/smoke_v010/smoke_result.json` (committed_steps=1,
398 observed sync calls) — COMPATIBILITY GATE SMOKE PASSED on the release
commit.

**Bounded off-policy online** (`artifacts/v0.1.0/bounded/bounded_online.json`,
design doc §9.2): on a real server rollout —
lag 1 ≤ bound 2 with declared correction → ALLOW; lag 5 > bound →
reject P005_LAG_EXCEEDS_BOUND; bounded mode without declared correction →
reject P006_CORRECTION_UNDECLARED.  3/3 matched.

**Canary sketch fix (2026-08-22)**: a bug in `canary.py`'s sketch made the
online canary-mismatch check read dict KEYS instead of token ids when the
generator returned TRL's dict directly (constant sketch ⇒ drift always 0).
Fixed with a regression test.  NOTE: the Day 2 loop's canary calibration and
v1 check used the `_unpack_gen` wrapper (tuple path) and were NOT affected;
the fix only affects direct-dict callers (the online mismatch check).

**P008 canary-mismatch online** (`artifacts/v0.1.0/canary/canary_mismatch_online.json`):
baseline greedy sketch on the REAL v0 weights; a deterministically perturbed
checkpoint (seed 7) loaded into the server and re-sketch →
**mismatch, drift=32 tokens** → the validator rejects with
`P008_CANARY_MISMATCH` (fail closed).

**Bounded off-policy ONLINE closed loop** (`artifacts/v0.1.0/second_wave/second_wave.json`,
design doc §9.2, decision D10): v0 trajectories consumed with lag=1 under
the bounded protocol (in-bound + declared correction → ALLOW), ONE
committed update (8 rollouts, ratio p50 1.0 / max 1.18), **398 observed
sync params**, v1 committed.  The committed manifest hash equals the Day-2
v1 hash — expected, since both commits re-serialize the same v0≈ weights
(loss≈0).  F1 gradient impact reported honestly as `undefined_near_zero`:
F1 is a CONTRACT fault (policy lag), the tensors are identical, and the
guard stops the stale consumption at validation (P004) before any optimizer
step — no fabricated cosine (design doc §12.3).

**Batch online matrix** (`artifacts/v0.1.0/batch_online/batch_online_matrix.json`,
decision D9): 32 REAL rollouts (8 prompts × 4 gens), every fault family
injected into EVERY generation —

| family | across generations |
|---|---|
| F1-F4 | 128/128 reject (32/32 per family) |
| F5-F8 | 128/128 reject/quarantine (32/32 per family) |
| normal set | 32/32 ALLOW |
| validator | 0.59 ms/envelope mean (0.55-0.98, n=32) |

**64-rollout batch online matrix** (`artifacts/v0.1.0/batch_online_64/`,
16 prompts × 4 gens): normal **64/64 ALLOW**; F1-F4 **256/256 reject**;
F5-F8 **256/256 reject/quarantine**; validator 0.63 ms/env mean (0.59-0.87,
n=64).

**128-rollout batch online matrix** (`artifacts/v0.1.0/batch_online_128/`,
32 prompts × 4 gens, decision D11): normal **128/128 ALLOW**; F1-F4
**512/512 reject**; F5-F8 **512/512 reject/quarantine** — 1024 fault
decisions with zero misses; validator 0.76 ms/env mean (0.73-1.04, n=128).
The per-family results are consistent across the 32-, 64- and 128-rollout
batches.

**256-rollout batch online matrix** (`artifacts/v0.1.0/batch_online_256/`,
64 prompts × 4 gens, decision D13 — the largest real-load batch on one
session): normal **256/256 ALLOW**; F1-F4 **1024/1024 reject**; F5-F8
**1024/1024 reject/quarantine** — 2048 fault decisions with zero misses;
validator 1.02 ms/env mean (0.98-1.31, n=256).  Ran on GPU0 with
`CUDA_VISIBLE_DEVICES=0` (GPU1 held another project's vLLM instance;
parallel-with recorded here, not in a manifest — this experiment's
evidence is the matrix json itself).

**Canary stress — determinism under load** (`artifacts/v0.1.0/canary_stress/`,
decision D11): 8 canary prompts (incl. a near-max-context prompt, ~900 words)
× 32 greedy tokens × **10 repeated sketches** against the real server in one
run: **max drift = 0 across all 10 repeats → deterministic**.  A nonzero
drift under repeated load would indicate environment nondeterminism
(e.g. shared-card interference); the fixed weight set is bit-stable.

**Full paired gradient replay** (`artifacts/v0.1.0/replay_all/gradient_replay.json`):
ALL 8 prompt groups × F2/F3/F4 = **24 pairs** (real model, v1 weights +
deterministic drift seed 7 σ=0.005).  Gradient-cosine distribution:

| family | n | cos mean | cos min | cos max |
|---|---|---|---|---|
| F2 misbound logprobs | 8 | 0.927 | 0.786 | 0.998 |
| F3 re-encoded tokens | 8 | 0.532 | 0.250 | 0.634 |
| F4 mask shift | 8 | 0.192 | **−0.029** | 0.431 |

overall relative L2 mean 1.78 (0.08-7.09).  The distributions quantify what
values cannot show: value-close misbound logprobs barely move gradients
(cos ≈ 0.93) while token/mask faults shift or flip the update direction.

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
