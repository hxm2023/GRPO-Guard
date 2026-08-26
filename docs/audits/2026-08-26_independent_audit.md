# GRPO-Guard E1/E2/F1 Independent Audit (2026-08-26)

**Auditor**: fresh-context reviewer (no prior project context; read-only)
**Snapshot**: main @ 3e37e1e
**Scope**: E1 fault survival, E2 non-inferiority, F1 static-rollout e2e evidence

## Findings → fixes

| # | Finding | Severity | Fix |
|---|---|---|---|
| 1 | **E1 guard-off arm never consumed the tampered batch**: `_prepare_inputs` returned the CLEAN `inputs`; the hook's tampered copy was only used for verification. On/off arms had byte-identical losses (same seed, step 10 both 0.4513) — the "corrupted updates applied 4/4" counterfactual was EMPTY | **FAIL (core)** | mixin now returns `to_verify` when `guard_enabled=False`; reran E1 2 seeds × on/off — on/off losses now diverge from the first fault step (19/30 steps differ), bad_updates off=2 real |
| 2 | `detection_latency_steps: 0` was hardcoded for the on arm | WARN | measured: min(block_step - fault_step) |
| 3 | runtime_attest absolute verdict threshold 1e-2 vs measured same-weights noise ~0.06-0.08 — every attestation (incl. healthy starts) was labelled STALE_RUNTIME_SUSPECTED, contradicting the E1 manifest's "CONSISTENT" | WARN | threshold → 0.15; the runners' relative verdict (late drift vs baseline) remains the primary stale flag |
| 4 | F1 `stale_detected` included `frozen_sync_calls > 0` — the injector's own counter as a self-witness | WARN | verdict is now drift-only (both seeds still flag: 0.241 vs 2×0.063, 0.176 vs 2×0.074) |
| 5 | E1 manifest overhead numbers stale vs the pack JSONs | WARN | manifest updated from the pack |
| 6 | E1 pack event stores hold the per-generation groups only | INFO | expected (30 steps = 8 generation windows); documented |

## Verdicts after fixes

- E1 on arm: PASS (0/4 bad updates, 0 latency — F3→T001, F2→L004 blocked before backward, losses unchanged)
- E1 off arm: PASS after fix (4/4 corrupted updates applied, 30 wasted steps; on/off losses diverge from the first fault)
- E2: PASS (disjoint held-out, per_q verified, margin test honest)
- F1: PASS (drift-only verdict, both seeds flag; seed-2 weak-training signal disclosed)
- General: no fabrication found; run-pack SHA256SUMS verifiable; claims within evidence

## Audit trail

- commit `3e37e1e`: pre-audit state
- fix commit: `mixin guard-off consumes tampered batch + measured latency + attestation threshold 0.15 + F1 drift-only verdict`
- rerun: E1 2 seeds × on/off with the fixed mixin (run_packs/e1_fault_survival updated)
