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

## D10 — Second-wave GPU experiments (2026-08-23, user-directed)

- **Decision**: spend remaining GPU budget (~25/80 GPU·h) on: (1) a SECOND
  guarded closed loop (v1 -> v2, two consecutive committed updates — shows
  repeatability); (2) a bounded off-policy ONLINE closed loop (lag=1
  consuming v0 trajectories to commit v1 — the full §9.2 path); (3) F1
  online gradient impact (gradients of a stale trajectory being silently
  consumed); (4) a 64-rollout online matrix (16 prompts x 4 gens); (5) a
  LoRA-variant closed loop if budget allows (base/adapter hashed
  separately).
- **Evidence**: all results machine-readable with run_id, committed under
  artifacts/, honest reporting (every pair/step reported).
- **Alternative**: stop at the current v0.2.0 — rejected: the user directed
  further GPU experiments for stronger resume evidence.
- **Why rejected others**: multi-machine consensus and cryptographic
  tamper-resistance remain explicitly out of scope (design doc §5.3, §7.3.4).
- **Falsification**: if the second loop diverges from the first (e.g. canary
  or validation fails), the discrepancy is investigated and disclosed
  before any release claim.

## D11 — Further real-load experiments (2026-08-23, user-directed)

- **Decision**: run a 128-rollout online matrix (32 prompts x 4 gens, the
  largest real-load batch on the same server setup) and a canary stress
  check (more prompts + longer greedy sketches), staying within the 80
  GPU·h cap (~17 h remain).  Extends D9's batch methodology; all
  pre-registered expectations unchanged.
- **Evidence**: machine-readable JSON with run_id, committed under
  artifacts/; honest reporting (every generation reported).
- **Falsification**: if the 128-rollout results diverge from the 64-rollout
  results (any family stops firing), the claim is narrowed and the
  discrepancy investigated before release claims.

## D12 — F5-F8 upgraded from v0.2-preview to formal v0.2 families (2026-08-23, user-approved)

- **Decision**: F5-F8 (split leakage / evaluator alias / event reorder /
  artifact mutation) become FORMAL v0.2 fault families.  The design doc
  §11 upgrade condition — "v0.2 冻结完整 injection protocol 后才升级为
  正式 fault families" — is met: `docs/INJECTION_PROTOCOL_v02.md` is frozen
  and the families pass every pre-registered expectation across all three
  matrix levels (frozen variants 12/12, online 4/4, batch online 256/256 +
  512/512 + 512/512, normal 4/4 + 32/32 + 64/64 + 128/128 ALLOW).
- **Evidence**: docs/INJECTION_PROTOCOL_v02.md; tests/frozen/f5_f8_v02/;
  artifacts/v0.2.0-dev/fault_matrix.json, fault_matrix_online.json;
  artifacts/v0.1.0/batch_online*/batch_online_matrix.json.
- **Alternative**: keep the preview label — rejected: the formalization
  condition in the design doc is satisfied, and the preview label
  understates completed, evidence-backed work on the resume.
- **Why rejected others**: no protocol or schema change is required — the
  formalization is a scope/status change only; v0.2.0-dev artifact paths
  stay as-is to preserve SHA256SUMS/CI integrity.
- **Falsification**: if any F5-F8 case is later found mis-classified
  (missed fault or false reject), the formal status is withdrawn and the
  discrepancy is disclosed before any further claim.

## D13 — Final real-load experiment: 256-rollout batch matrix (2026-08-23, user-approved)

- **Decision**: run the largest real-load batch online matrix — 256 REAL
  rollouts (64 countdown prompts x 4 gens) with F1-F8 injected into EVERY
  generation (1024 F1-F4 decisions + 1024 F5-F8 decisions + 256 normal
  ALLOW checks), on autodl2 GPU1 sharing the card with agent-ttrl
  (recorded as parallel-with in the manifest; validator timing is CPU-side
  and not degraded).  Final GPU experiment under the 80 GPU·h cap
  (~66/80 used before this run; expected <= 3 GPU·h).
- **Evidence**: machine-readable JSON with run_id, committed under
  artifacts/v0.1.0/batch_online_256/; every generation reported; parallel
  load observed via nvidia-smi in the manifest.
- **Alternative**: stop at 128 rollouts — rejected: user directed a final
  larger real-load run; 256 is the largest practical batch on one session.
- **Why rejected others**: a second closed loop is forbidden by design doc
  §17 (ONE committed update-sync cycle); the 256 matrix is the maximum
  density of fault-decision evidence per GPU·h.
- **Falsification**: if any family stops firing or any normal generation is
  rejected, the claim is narrowed to the observed subset and the
  discrepancy is investigated before any release claim.

## D14 — Multi-step closed loop + context stress (2026-08-23, user-approved)

- **Decision**: extend the online evidence with (1) a MULTI-STEP guarded
  closed loop — 3 consecutive committed update-sync-rollout cycles on the
  real server (v0→v1→v2→v3), so the optimizer's weights genuinely move
  across steps (the v0.1 loop's single update consumed its own policy's
  trajectories, loss≈0; a multi-step run yields nonzero loss and a
  measurable weight delta between v0 and v3); (2) a longer max-context
  rollout (near the 2048 limit) to stress the mask/span reconstruction at
  the boundary.  Design doc §17 Day 2 locked ONE real online closed loop
  for the v0.1 release evidence — that evidence stands untouched; this is
  a post-release extension recorded here BEFORE the run, within the
  remaining ~14 GPU·h.
- **Evidence**: machine-readable JSON + PolicyManifests per step,
  committed under artifacts/; honest reporting (loss, ratio stats, weight
  deltas per step, parallel-with state).
- **Alternative**: stop at v0.1 evidence — rejected: the user directed
  deeper real-load evidence ("多步 GRPO 更新让权重真实移动").
- **Why rejected others**: a longer single loop (more rollouts per step)
  would burn budget for the same evidence class; multi-step is the only
  change that yields genuinely new evidence (nonzero loss + real weight
  movement + multi-cycle canary).
- **Falsification**: if any step fails validation/canary, the multi-step
  claim is narrowed to the steps that passed and the discrepancy is
  investigated before any release claim.

## D15 — Real off-policy RL training loop (2026-08-23, user-approved)

- **Decision**: run the framework's first REAL multi-step RL training loop
  on autodl2 — bounded off-policy (lag=1 via a one-step rollout buffer):
  each step's update consumes the PREVIOUS step's rollouts while the
  model has already advanced one update, so ratios deviate from 1, loss
  is nonzero, and the bf16 weights genuinely move (the D14 on-policy
  loop could not move weights — math constraint, recorded there).  ~30
  steps of GRPO on Qwen3-4B countdown, tracking success rate per step
  (v0 baseline ~25% from the Day 2 rewards), with the guard active on
  EVERY step (identity + pre-update ALLOW required, observed sync,
  canary check, committed manifest per step).  Budget: ~8-10 GPU·h of
  the remaining ~11.
- **Evidence**: rl_training.json with per-step loss/ratio/success/sync/
  canary + the training curve; committed under artifacts/; honest
  reporting (every step).
- **Alternative**: stop at the D14 multi-step loop — rejected: the user
  directed an experiment whose success-rate curve proves the framework
  guards REAL learning, the strongest interview signal.
- **Why rejected others**: on-policy repeats the D14 zero-delta result;
  a pure simulation would not produce real server rollouts.
- **Falsification**: if success rate does not improve over ~30 steps (or
  any step fails validation/canary), the claim is narrowed to the
  observed curve and the discrepancy is investigated before any claim.

## D16 — F9-F10 fault families (2026-08-23, user-approved)

- **Decision**: extend the fault matrix with two task-agnostic families —
  F9 reward injection (R008_REWARD_VERIFIER_UNREGISTERED: the reward event
  claims a verifier that is not in the registered evaluator registry —
  reject) and F10 data poisoning (D004_PROMPT_UNREGISTERED: a generation
  whose prompt is not in the frozen split manifest — reject).  Both are
  deterministic, require no tokenizer/verifier recompute (task-agnostic),
  and extend the design doc §11 family table via status notes.
- **Evidence**: rules + injectors + frozen fixtures + config +
  tests/frozen/f9_f10_v01/ matrix; all pre-registered expectations.
- **Alternative**: verifier-recompute detection (re-validate the reward
  against the actual completion) — rejected: it needs per-task verifiers
  and tokenizer access inside the validator, breaking task-agnosticism.
- **Why rejected others**: prompt-content hashing (detect text
  substitution) would couple the validator to text decoding; registry
  checks capture the same class of silent corruption cheaply.
- **Falsification**: if any F9/F10 fixture is mis-classified, the family
  claim is withdrawn and the discrepancy investigated.
