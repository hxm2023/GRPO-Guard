# Changelog

## v0.4.0-dev (2026-08-25, post-audit)

Evidence audit (`docs/audits/2026-08-25_*`) drove five P0 fixes:

- **P0-1 crash-consistent update** — `guarded_optimizer_step` now records a
  WAL lifecycle (`PREPARED → APPLIED → CHECKPOINTED → COMMITTED / ABORTED`),
  exposes failpoints for loss/backward/step/checkpoint/commit, and provides
  atomic checkpoint promotion (fsync shards + atomic rename).  Recovery
  semantics are documented as "discard worker, resume from last committed
  checkpoint" — in-memory parameter rollback is explicitly NOT claimed.
- **P0-2 transactional nonce** — `NonceRegistry` moved from append-only
  JSONL to SQLite (`UNIQUE(nonce)`, `BEGIN IMMEDIATE`), with a 32-process
  race test proving exactly-once consumption; legacy JSONL registries are
  imported automatically on first open.
- **P0-3 actual-input binding** — `materialize` now freezes the reward
  value hash/shape/dtype, GRPO group size, group membership order and the
  expected parent policy manifest into the sealed `UpdateInputEvent`;
  `single_use_nonce_sha256` stores a real SHA-256; external `loss_fn`
  injection was removed from the production entry.  Negative tests
  (reward substitution, group reorder, wrong group size, wrong model
  object) all fail before backward.
- **P0-4 official-trainer actual-input verification** — `GuardedGRPOTrainer`
  records REAL mask/logprob/reward artifacts (no zero-hash placeholders)
  and verifies the ACTUAL `training_step` tensors (`input_ids` rows, old
  logprob completion spans, advantages row count) before
  `super().training_step`; rollout records rotate per step.
- **P0-5 release governance** — version marked `0.4.0-dev`; new frozen
  evidence pack `artifacts/v0.4.0/` with `RELEASE_MANIFEST.json` and a real
  provenance `run_manifest.json`; `artifacts/v0.2.0-dev` checksums
  recomputed to the current bytes; CI verifies ALL published packs and
  uploads the coverage report; README strong claims ("atomic",
  "unbypassable") replaced by an exact capability table.

New docs: `docs/CLAIM_MATRIX.md` (claim → evidence → status),
`docs/audits/` (full audit trail).

## v0.3.0 (2026-08-24)

- GuardedGRPOTrainer (P1-1): official TRL GRPOTrainer wrapped at
  rollout/step/commit seams; 1-step real server-mode smoke.
- P1-2: guard on/off 3-seed comparison + frozen held-out probe
  (25% → 25%, honest negative).
- P1-3: tool-use trajectory contract (`tool_env.py`).
- `grpo-guard` CLI: verify / resume / metrics / doctor / alert-scan.
- 20-step P0-fixed RL training with 3× interruption/resume; stale-runtime
  detection experiment.

## v0.2.1 (2026-08-24)

- F9 reward injection (`R008_REWARD_VERIFIER_UNREGISTERED`) and F10 prompt
  poisoning (`D004_PROMPT_CONTENT_MISMATCH`); frozen 3/3 + normal 4/4.

## v0.2.0 (2026-08-23)

- F5–F8 formal fault families (split leakage / evaluator alias / event
  reorder / artifact mutation); injection protocol frozen; 12/12 variant
  matrix; P008 canary-mismatch online reject.

## v0.1.0 (2026-08-22)

- Initial release: schema/event store, staged validator, guarded
  single-use update handle, paired replay, Day 1–5 gates on Qwen3-4B +
  TRL/vLLM server (autodl2, 2×RTX 6000D).
