# GRPO-Guard Decision Log

Every design deviation recorded BEFORE the first formal run, never after
seeing results (CLAUDE.md honesty rules).  Each entry: Decision · Evidence ·
Alternative · Why rejected · Falsification.

## D1 — Compatibility matrix (Day 1, 2026-08-22)

- **Decision**: pin autodl2 observed matrix: python 3.12.3, torch 2.11.0+cu130,
  transformers 5.15.1, trl 1.10.0, vllm 0.26.0, accelerate 1.14.0, Qwen3-4B
  @ 1cfa9a7208912126459214e8b04321603b3df60c (`compatibility_profile.yaml`).
- **Evidence**: the design doc's candidate versions were "candidate starting
  points" (§4.1.1); the box resolved torch 2.11.0+cu130 through vllm 0.26.0's
  requirement.  The whole stack imports together and the official smoke
  passed on it.
- **Alternative**: pin torch 2.8.0+cu128 from the base env — rejected because
  vllm 0.26.0 pulled torch 2.11.0 in the clean venv and the base env is not
  the project env.
- **Falsification**: any future run that fails on this matrix regresses the
  freeze.

## D2 — Adapter patches for trl 1.10.0 + vllm 0.26.0 (Day 1)

- **Decision**: two small, version-guarded, fail-closed adapter patches:
  (a) `VLLMClient.init_communicator` normalizes the unindexed
  `torch.device('cuda')` to the current device before vLLM's
  `PyNcclCommunicator` init (whose warm-up all_reduce asserts
  `in_tensor.device == self.device`); (b) remove torchcodec (broken .so,
  no reverse dependencies).
- **Evidence**: unpatched, the trainer crashed in pynccl init before any
  rollout; after patching, the full smoke passed (1 step + 398 sync calls).
- **Alternative**: downgrade vllm to 0.25.x — rejected: larger env churn,
  unverified behavior, and the patch is 10 lines per design doc §14.2.
- **Falsification**: if TRL upstream fixes the device handling, the version
  guard asserts and the patch fails closed (smoke stops loudly).

## D3 — Rollout driven by VLLMClient.generate, not trainer.train() (Day 2)

- **Decision**: the guarded closed loop drives rollout through TRL's own
  `VLLMClient.generate`, which returns the server's authoritative
  `prompt_ids`, `completion_ids` and per-token service logprobs; the runtime
  adapter emits GenerationEvents from those bytes.  The optimizer step is
  executed by the Guard's update adapter on ValidatedBatchHandle tensors.
- **Evidence**: TRL 1.10's internal training path re-tokenizes completions
  for its own loss — by construction the F3 risk the Guard exists to catch;
  `trainer.train()` would also double-apply an optimizer step.
- **Alternative**: monkeypatch TRL's `compute_loss` — rejected: fragile
  against upstream, and the design's guarded-update contract (§7.3.3) makes
  the adapter the only consumer of validated tensors.
- **Falsification**: if the guarded update cannot produce a real committed
  optimizer step on the authoritative tensors, the Day 2 loop fails and the
  project degrades per design doc §7.3.3.

## D4 — v0 checkpoint manifest from original files (Day 2)

- **Decision**: the v0 PolicyManifest is computed by hashing the original
  checkpoint shards on disk (no re-save); the v1 manifest is computed from
  the freshly saved safetensors shards.
- **Evidence**: the v0 weights ARE the model files; re-loading the model just
  to re-save identical bytes wastes a GPU load for no new evidence.
- **Falsification**: if the server's loaded v0 differs from the hashed files
  (corrupt cache), the canary calibration catches it (mismatch → abort).

## D5 — Canary tolerance (Day 2)

- **Decision**: tolerance frozen from the max token-level drift observed over
  5 reloads of the same checkpoint; greedy sketches only; canary windows hold
  the shared lock file (shared-card rule 1).
- **Evidence**: design doc §10 requires ≥5 reloads; v0.1 canary is a behavior
  sketch, not a byte proof (§5.3).
- **Falsification**: if reload drift is so large that v1 never passes, the
  canary is downgraded to auxiliary evidence (design doc §21 risk table).

## D6 — Warm-up identity check for the trainer model (Day 2)

- **Decision**: the trainer model on GPU0 loads the same checkpoint files the
  server serves; identity is attested by the shared checkpoint manifest hash
  (not by comparing GPU tensors).
- **Evidence**: same files → same weights; manifest hash is the control-plane
  layer of the two-layer evidence (§10).
- **Falsification**: if a stale HF cache served different weights, the canary
  mismatch or manifest hash mismatch aborts the loop.

## D7 — v0.2-preview extension: F5-F8 fault families (2026-08-22)

- **Decision**: implement F5 (split leakage → `D003_SPLIT_OVERLAP`, reject +
  overlap report), F6 (evaluator alias → `R006_EVALUATOR_ALIAS`, quarantine),
  F7 (event reorder → L005/P007 with explicit fixtures), F8 (artifact
  mutation → T001 with isolated-store injection) as **v0.2-preview** —
  explicitly NOT part of the v0.1 matrix (design doc §11: F7/F8 upgrade to
  formal families only in v0.2 with a frozen injection protocol).
- **Evidence**: v0.2 matrix over the REAL loop artifacts: 4/4 matched
  (pre-registered expectations in configs/faults/f5_f8_v02.yaml), normal
  4/4 allow; frozen fixtures under tests/frozen/f5_f8_v02 with
  no-overwrite; unit tests green; full suite green (126 tests).
- **Alternative**: leave F5-F8 unimplemented until v0.2 — rejected: the
  user asked to continue upgrading; the rule + fixture work is CPU-only and
  de-risks the v0.2 freeze.
- **Why rejected others**: no GPU budget spent; v0.1 release (tag v0.1.0)
  is untouched and remains the authoritative gate-passed release.
- **Falsification**: if the v0.2 matrix cannot run against real loop
  artifacts or the fixtures drift from pre-registered expectations, the
  v0.2-preview claim is withdrawn.

## D8 — v0.2-preview demo artifact (2026-08-22)

- **Decision**: add a CPU-only 3-5 minute demo (`examples/countdown/demo.py`
  + `docs/demo.md`) covering happy path, F1-F4, F5-F8, and the fail-closed
  guarded update.
- **Evidence**: demo output verifies all eight fault families + text-input
  refusal in ~3 minutes, no GPU required (design doc §23 demo requirement).
- **Falsification**: if the demo output diverges from the gate results, the
  demo is fixed or removed before any claim uses it.

## D9 — Batch online experiments on autodl2 (2026-08-22, user-directed)

- **Decision**: run batch online experiments on the rented autodl2 GPUs:
  (1) F1-F8 online matrix over 32 REAL rollouts (8 prompts × 4 gens) with
  per-family multi-generation injection and normal-set 32/32 ALLOW plus
  online validator timing; (2) paired gradient replay over ALL 8 prompt
  groups × F2/F3/F4 (24 pairs) for metric distributions instead of single
  points.
- **Evidence**: pre-registered expectations from configs/faults/*.yaml;
  results written as machine-readable JSON with run_id and committed under
  artifacts/; honest reporting (no cherry-picking — every pair reported).
- **Alternative**: keep the single-point online matrices — rejected: the
  user directed batch-scale experiments, and batch statistics are stronger
  evidence than single points.
- **Why rejected others**: the closed loop is intentionally ONE committed
  update-sync cycle (design doc §17); a second loop would burn budget for
  no new evidence class.  Budget check: ~40/80 GPU·h used, ~40 remain.
- **Falsification**: if batch results contradict the single-point results
  (e.g. a fault family stops firing on some generation), the matrix claim
  is narrowed to the observed subset and the discrepancy is investigated
  before any release claim.
