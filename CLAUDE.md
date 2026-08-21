# GRPO-Guard — Trajectory Contract, Lineage & Fault-Injection Framework (Engineering Project)

**Goal**: Build GRPO-Guard v0.1 — a machine-verifiable evidence chain between online
LLM post-training components (trainer / rollout / behavior scoring / reward /
optimizer update), detecting silent errors (static rollout policy, misbound
old-logprob, retokenization, mask shift). This is an ENGINEERING flagship project
(job-market: post-training infra / agent RL infra / eval reliability), NOT a paper
project. "Accuracy improvement" is not the goal; "proving training consumes the
correct trajectories" is.

**AUTHORITATIVE REQUIREMENTS**: `GRPO-Guard_详细项目设计与旧项目迁移手册.md` (project
root, v1.0 2026-08-22) is the single source of truth for design, schema, protocols,
faults, gates, tests, budget and release — this CLAUDE.md adds ARIS execution
discipline on top; it does NOT redesign the project. Read the design doc FIRST and
re-read it at every gate.

## What's Locked (from the design doc — non-negotiable)

- **Clean new repo** (do not inherit grpo-credit-assignment's .git; old repo is
  legacy-evidence only, never rename-and-continue).
- **Fixed workload**: Qwen/Qwen2.5-1.5B-Instruct (fixed revision), Countdown task,
  frozen train/eval manifests, deterministic rule verifier, TRL GRPO + vLLM server,
  GPU0 trainer / GPU1 rollout, BF16 preferred, ≥1 real committed optimizer update.
- **Compatibility matrix frozen BEFORE work** (`compatibility_profile.yaml`); the
  official server-mode GRPO smoke must pass before F1 online experiments.
- **F1-F4 canonical faults** (static rollout / misbound old-logprob / retokenization /
  mask shift) with normal + boundary + held-out variants; F5-F8 deferred to v0.2.
- **Producer ownership + event lineage + content hashing** (EventBase, PolicyManifest,
  GenerationEvent, ScoringEvent, RewardEvent, TrajectoryEnvelope, ValidationDecision;
  canonical JSON, SHA-256, append-only store, single-writer lease + fencing).
- **Guarded update**: only a single-use ValidatedBatchHandle; no text fallback, no
  re-tokenization after validation.
- **Reason-coded validation** (P/T/M/L/D/R rule tables) with allow/quarantine/reject.
- **Deterministic paired replay** (frozen producer artifacts; gradient cosine /
  relative L2 / update norm / ratio / clip metrics; report undefined_near_zero when
  norms ≈ 0).
- **Three gates** (Day 3 Correctness / Day 4 Impact+Overhead / Day 5 Release) with the
  exact pass conditions in the design doc §16. Quarantine/reject never enters
  reward/update consumption.
- **Budget**: 30-36 A800 GPU·h hard cap 40 (2×A800); light artifacts <5 GB;
  never publish full checkpoints.
- **Forbidden legacy narratives** (design doc §2.2, §3.2): success-rate curves,
  static-policy claims, ρ=0.735, "CPC works", re-used production trainer code as-is.
  Killed legacy logic may only be rebuilt as minimal fault fixtures marked
  `reconstructed_from_incident`.
- **Definition of Done** = design doc §23 checklist; resume bullets only from
  gate-passed numbers (§20).

## ARIS Role (adapted for an engineering project)

- **No Phase 0-1 re-discovery**: the design is authoritative and locked. ARIS
  contributes: execution discipline (gate checks, decision logs), review loops
  (code/design review at each gate — reviewer fallback per shared references),
  experiment management (run manifests, honest metrics), and release rigor.
- **Review gates**: before each of the three gates, run an independent review
  (cross-model per `shared-references/reviewer-fallback.md`; images read natively by
  the model) targeting: contract-vs-implementation consistency, fail-closed behavior,
  test completeness, claim honesty (no overclaiming detection rates).
- **Decision log**: every design deviation (e.g., model substitution, adapter
  differences) recorded with Decision · Evidence · Alternative · Why rejected ·
  Falsification — before first formal run, never after seeing results.
- **Honesty rules** (inherited): no post-hoc gate loosening (reject→quarantine to
  pass counts is forbidden), frozen fixtures never auto-overwritten, all published
  numbers traceable to artifact + commit + checksum.

## Compute (server TBD — reserved, do not bind)

- Server choice is PENDING (candidates: jindun 8×A800, autodl1 2×RTX 6000D, local
  RTX 5060 8GB for CPU/light parts). When chosen, record in this file; GPU
  contention rules of the chosen box apply (other users first).
- Budget regardless of server: 30-36 GPU·h hard cap 40, 2 GPUs enough, wall time
  15-20h segmented. Checkpoint/resume default; save progress before sacrificing a run.

## Pipeline (adapted)

- Phase 0: re-read design doc + old-repo asset index (§22) + compatibility matrix
  freeze (already-planned; no literature discovery needed).
- Phase 1: frozen design confirmation + Gate pre-checks + environment bring-up
  (official TRL+vLLM server smoke).
- Phase 2 (engineering loop): Day1-Day5 plan execution with per-day review;
  gates at Day 3/4/5; every run produces `run_manifest.json + REPORT.md +
  SHA256SUMS` (no-overwrite canonical outputs).
- Phase 3: release package (README, demo, report, checksums, LICENSE Apache-2.0
  where feasible), resume bullets per design doc §20 status ladder.
- Compliance: human owns the project; AI participation disclosed where required.
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 108 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/c/Users/w1828/repos/aris_repo`).
Update with: `bash /c/Users/w1828/repos/aris_repo/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
