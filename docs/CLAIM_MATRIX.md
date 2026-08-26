# GRPO-Guard Claim Matrix

Every published/resume claim, its evidence tier, the artifact that supports
it, and its honest status.  Rule (design doc §20, audit §13): resume and
README only use numbers traceable to `artifacts/v0.4.0/` + commit +
SHA256SUMS, with the status below.

**Evidence tiers**
- T0: unit/synthetic (CPU tests, frozen fixtures)
- T1: real rollout + injected fault (Qwen3-4B / vLLM server, autodl2)
- T2: custom real update loop (guarded_optimizer_step, WAL, nonce)
- T3: official TRL GRPOTrainer guarded path (seam instrumentation; full
  update-input gating pending GPU run)

| # | Claim | Tier | Evidence (artifacts/v0.4.0/) | Status |
|---|---|---|---|---|
| C1 | 256 real rollouts, F1–F8 injected per trajectory, 2048 decisions match frozen oracle | T1 | `batch_online_256/batch_online_matrix.json`, `fault_matrix.json` | supported, **must say "256 real rollouts + targeted injected decisions"**, NOT "2048 real incidents" |
| C2 | 512 rollouts × 8 = 4096 decisions | T1 | `batch_online_512/batch_online_matrix.json` | supported (main snapshot; same qualifier as C1) |
| C3 | normal trajectories 256/256 ALLOW, 0 false reject | T1 | `batch_online_256/`, `batch_online_64/` | supported |
| C4 | 24 paired gradient probes quantify F2/F3/F4 (cos 0.93/0.53/0.19) | T1 | `replay/gradient_replay.json`, `day4_summary.json` | supported with qualifier: **v1 weights + deterministic drift(seed=7,σ=0.005), mechanism probe** |
| C5 | 32/32 identity + 32/32 pre-update ALLOW; 1 real committed step; 398 param sync; canary v1 pass | T1/T2 | `loop/`, `smoke/`, `smoke_v010/` | supported |
| C6 | bounded off-policy GRPO 20 steps: non-zero loss, weight delta 9.53–10.4 fp32, ALLOW every step, 3× interruption/resume | T2 | `rl_training_final/`, `rl_training/` | supported (in-train evidence; NOT a capability claim) |
| C7 | guard overhead 0.59–0.63 ms/envelope; 40.4 ms/batch offline | T1 | `overhead.json`, `guard_compare/` | supported (measured with observed GPU util recorded) |
| C8 | stale-runtime detection (server v0 vs trained v20, drift 7 → DETECTED) | T2 | `sync_noop/sync_noop_result.json` | supported |
| C9 | guard on/off 3 seeds: 0.698±0.049 vs 0.709±0.050 | T2 | `guard_seeds/` | supported as **"preliminary no obvious degradation in small smoke"**, NOT statistical equivalence |
| C10 | held-out 8 frozen prompts 25%→25% (no generalization) | T2 | `heldout/heldout_eval.json` | supported — deliberate negative result |
| C11 | GSM8K 28%→78% capability improvement | — | — | **unsupported — never claim**; in-train curve peak only, collapse at end |
| C12 | precondition failures (non-ALLOW, tampered artifact, reward substitution, wrong group/model, nonce reuse, tokenizer call) reject BEFORE backward, params unchanged | T0/T2 | tests/unit/test_guarded_step.py, guarded_update.py | supported + tested |
| C13 | transactional exactly-once nonce across processes | T0 | tests/unit/test_guarded_step.py (32-process race) | supported (Linux CI) |
| C14 | crash-consistent update lifecycle (WAL PREPARED→COMMITTED/ABORTED), atomic checkpoint promotion | T0/T2 | guarded_update.py, test_guarded_step.py failpoints | supported; **in-memory rollback NOT claimed** |
| C15 | actual reward/group-size/group-order/model identity bound to handle | T0 | guarded_update.py, test_guarded_step.py | supported (v0.4.0) |
| C16 | official GuardedGRPOTrainer verifies actual step inputs (tokens/logprobs/advantages) | T0+T3 | guarded_grpo_trainer.py, test_guarded_grpo_trainer.py, `run_packs/p0_4_official_trl/` | supported — CPU tests + **20-step official-path GPU run** (`ok=true`, per-step verified) |
| C23 | official-path optimizer.step() capability-gated (no capability → GuardViolation) | T0 | guarded_grpo_trainer.py (_CapabilityOptimizer), test_guarded_grpo_trainer.py | supported + tested |
| C24 | dual-source runtime attestation (server /get_sequence_logprobs fingerprint vs trainer forward; STALE_RUNTIME_SUSPECTED on drift) | T0 | runtime_attest.py, test_runtime_attest.py | supported at unit level; **GPU-run wiring in E1** (start/end attestation) |
| C25 | E1 fault-injected training survival (guard on/off × 2 seeds): bad update accepted, detection latency, wasted steps | T3 | `run_packs/e1_fault_survival/`, examples/countdown/fault_survival.py | supported — guard-on: **0/4 bad updates, 0-step latency, 0 wasted steps** (F3→T001, F2→L004 blocked before backward); guard-off: 4/4 bad updates, 30 wasted steps; ~1% step-time overhead; attestation CONSISTENT all runs |
| C26 | E2 guard on/off non-inferiority (5 seeds × 2 arms, 16-prompt frozen held-out, pre-registered margin ≤1/16) | T3 | `run_packs/e2_non_inferiority/` | supported — held-out on 0.250±0.06 vs off 0.288±0.11: delta -0.0375 **within margin**; on-arm variance smaller; train reward on 0.273 vs off 0.263; step +2.3% |
| C17 | official trainer runs: 1-step smoke + 20-step with F3 injection blocked | T3 | `guarded_trainer/guarded_trainer_smoke.json`, `run_packs/p0_4_official_trl/guarded_trainer_official_run.json` | supported — 20 steps verified; faults at steps 10/19 blocked before backward (T001), recovered; commit sha recorded |
| C18 | TRL upstream PR #6876 accepted/merged | — | github.com/huggingface/trl/pull/6876 | **unsupported — "open PR" only** |
| C19 | core coverage 87% | — | CI artifacts (coverage.xml) | supported as "gate ≥80%"; exact % downloadable from CI artifact |
| C20 | current main == released v0.3.0 | — | git tags | **unsupported** — main is v0.4.0-dev; frozen pack = artifacts/v0.4.0 |
| C21 | immutable historical packs | — | RELEASE_MANIFEST.json | v0.4.0 frozen; **v0.1.0 was historically appended to (documented gap)** |
| C22 | production-ready / multi-node / NPU / malicious-producer resistance | — | — | **unsupported — out of scope** (single-box 2-GPU prototype; detects dev-env wiring errors) |

## Claim wording rules (audit §9)

- "atomic optimizer transaction" → "preconditions fail before backward;
  post-step failures follow crash-recovery (WAL); no in-memory rollback"
- "unbypassable single entry" → "capability-gated optimizer entry"
- "2048/4096 real incidents" → "256/512 real rollouts + targeted injected
  decisions matching a frozen oracle"
- "GSM8K improvement" → "in-train reward curve with collapse; frozen
  held-out probe 25%→25% (no generalization)"
- "official GRPOTrainer guarded" → "seam instrumentation + CPU-level
  actual-input verification; full GPU-path run pending"
- "coverage 87%" → "coverage gate ≥80% (report downloadable in CI)"
