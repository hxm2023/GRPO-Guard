# GRPO-Guard

[![CI](https://img.shields.io/github/actions/workflow/status/hxm2023/GRPO-Guard/ci.yml?branch=main&label=CI)](https://github.com/hxm2023/GRPO-Guard/actions)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)

Trajectory contract, lineage and fault-injection framework for online LLM
post-training (TRL GRPO + vLLM server).

GRPO-Guard builds a **machine-verifiable evidence chain** between the online
training components — trainer / rollout / behavior scoring / reward /
optimizer update — and uses it to detect the silent errors that training
curves cannot reveal:

1. **static rollout policy** — the runtime keeps serving an old policy while
   the trainer updates its weights;
2. **misbound old-logprob** — "old" log-probs that were not produced by the
   behavior policy that generated the trajectory;
3. **retokenization** — the sequence the trainer optimizes is not the
   sequence the server sampled;
4. **mask shift** — completion/action masks that select prompt or padding
   tokens.

It does not reimplement TRL, vLLM or GRPO: it wraps them with content-addressed
artifacts, append-only events, reason-coded validation and a single-use
guarded update handle.

> **Status: v0.3.0 (released).** All three gates (Compatibility /
> Correctness / Impact+Overhead) have passed on autodl2 (2×RTX 6000D) with
> Qwen/Qwen3-4B.  The authoritative design is
> `GRPO-Guard_详细项目设计与旧项目迁移手册.md` (v1.0).  Every number in this
> README traces to `artifacts/v0.1.0/` + commit + SHA256SUMS.  This is a
> production-oriented prototype: single-box 2-GPU validation, CPU CI, no
> multi-node/DDP runs yet (docs/PROJECT_INTRO.md honest boundaries).

## Architecture

```
Dataset and Split Manifest → TRL GRPO Trainer
  → committed policy v+1 → Checkpoint + PolicyManifest (content-hashed)
  → upstream sync (observed update_named_param) → vLLM Runtime Adapter
  → GenerationEvent + token artifacts (the server's OWN token ids)
  → Append-only Event/Artifact Store
  → Pre-reward Envelope (reference-only) → Identity Validator
     → ALLOW → Reward Adapter → RewardEvent
     → QUARANTINE/REJECT → reason-coded report (never consumed)
  → Pre-update Envelope → Pre-update Validator → ALLOW
  → Guarded Batch Materializer → single-use ValidatedBatchHandle
  → GRPO update (real optimizer step) → UpdateCommitted
  → canary check → v+1 rollout
```

Key invariants (design doc §6-§9):

- **Producer ownership**: the runtime adapter is the only producer of
  generation events and token artifacts; the control plane is the only
  producer of sync/update events; the materializer is the only producer of
  the update input.  No component may rewrite another's output.
- **No re-tokenization**: the optimizer consumes the exact token ids the
  server sampled (via TRL's `VLLMClient.generate` response), with canonical
  masks reconstructed from spans (design doc §7.9).  Text is a read-only
  view for the reward verifier.
- **Reason-coded validation**: P/T/M/L/D/R rule tables; a decision is
  `allow`, `quarantine` or `reject` with machine-readable codes
  (e.g. `P004_STALE_POLICY_STRICT`, `T002_TOKENIZER_MISMATCH`,
  `M004_CANONICAL_MASK_MISMATCH`, `L003_SCORER_POLICY_MISMATCH`).
  Only envelopes with **both** identity and pre-update `ALLOW` may update
  parameters; the guarded update adapter refuses anything else and any nonce
  reuse.
- **Deterministic paired replay**: fault pairs are derived from the same
  frozen producer artifacts; gradients are compared with cosine / relative
  L2 / update norm / ratio / clip metrics (`undefined_near_zero` when norms
  ≈ 0 — never fabricated 0).

## Quickstart

```bash
uv sync --frozen          # CPU contract deps (pydantic, numpy, pyyaml)
uv run pytest tests/      # unit + contract + property tests
```

GPU (server mode) requires the compatibility matrix in
`compatibility_profile.yaml` (autodl2: torch 2.11.0+cu130, trl 1.10.0,
vllm 0.26.0, transformers 5.15.1, Qwen3-4B).

```bash
uv run grpo-guard contract-check --cases tests/frozen/f1_f4_v01   # frozen matrix
uv run grpo-guard day3-matrix --loop-dir artifacts/v0.1.0/loop    # gate on real artifacts
uv run grpo-guard replay --manifest artifacts/v0.1.0/run_manifest.json
uv run grpo-guard report --artifact-dir artifacts/v0.1.0
```

## Single-fault demo

```bash
uv run python - <<'PY'
from grpo_guard import testing
from grpo_guard.faults import inject_f4_mask_shift
from grpo_guard.validators.context import ProtocolConfig, ValidationContext
from grpo_guard.validators.validator import validate_envelope

t = testing.build_trajectory()                      # canonical happy path
f = inject_f4_mask_shift(t, shift=1)                # shift the completion mask by 1
ctx = ValidationContext(envelope=f.envelope, store=f.store, events=f.events,
                        policy_manifest=f.policy_manifest, split_manifest=f.split_manifest,
                        protocol=ProtocolConfig(name="strict_v01", mode="strict_on_policy"))
d = validate_envelope(ctx, "identity_pre_reward").decision_payload
print(d.decision, d.reason_codes)                   # reject ['M002_PROMPT_SELECTED', 'M004_CANONICAL_MASK_MISMATCH']
PY
```

## Results (gate-passed, artifacts in `artifacts/v0.1.0/`)

| Gate | Result | Evidence |
|---|---|---|
| Day 1 Compatibility | TRL+vLLM server-mode smoke green; 1 committed step; **398 observed weight-sync calls** | `smoke/`, `compatibility_profile.yaml` |
| Day 2 Closed loop | v0 rollout → identity ALLOW 32/32 → pre-update ALLOW 32/32 → real update → sync → canary v1 pass → v1 rollout validated | `loop/` |
| Day 3 Correctness | canonical F1–F4 **4/4 reject** (pre-registered codes); 12/12 variants; normal **32/32 ALLOW** (0 false reject); boundary 4/4; stale acceptance 0 | `fault_matrix.json` |
| Day 4 Impact/Overhead | paired gradients: F2 cos 0.989, F3 cos 0.634, F4 cos 0.238 (real model); guard overhead 40.4 ms/batch (3 repeats); F1 update norm 0.0 measured | `replay/`, `overhead.json`, `day4_summary.json` |

## Online experiments (autodl2, real server rollouts)

All runs below executed on autodl2 (2xRTX 6000D) against the REAL vLLM
server (trl vllm-serve + Qwen3-4B), evidence committed under
`artifacts/v0.1.0/`:

| experiment | result | evidence |
|---|---|---|
| F1-F8 online matrix | F1-F4 4/4 reject, normal 4/4 ALLOW; F5-F8 4/4 (formal v0.2) | `online/fault_matrix_online.json`, `v0.2.0-dev/fault_matrix_online.json` |
| bounded off-policy online | lag≤bound+correction ALLOW; lag>bound reject P005; missing correction reject P006 — 3/3 | `bounded/bounded_online.json` |
| tag v0.1.0 clean smoke | 1 committed step, 398 sync calls (release commit reproduced) | `smoke_v010/smoke_result.json` |
| Day 2/5 guarded closed loop | 32/32 ALLOW → real update → 398-param sync → canary v1 pass → v1 rollout | `loop/` |
| Day 4 paired replay | F2 cos 0.989, F3 cos 0.634, F4 cos 0.238 (real model) | `replay/gradient_replay.json` |
| P008 canary mismatch | perturbed weights → canary drift 32 tokens → validator reject | `canary/canary_mismatch_online.json` |
| canary determinism stress | 10/10 repeated greedy sketches drift 0 (8 prompts incl. near-max-context) | `canary_stress/canary_stress.json` |
| 256-rollout batch matrix (D13) | 256 real rollouts: normal 256/256 ALLOW; F1-F8 injected per trajectory — 2048 decisions all matching the frozen oracle | `batch_online_256/batch_online_matrix.json` |
| multi-step closed loop (D14) | 3× committed update-sync-rollout (v0→v3), 3×398-param sync, 3× canary pass, 1876-token boundary rollout ALLOW | `multi_step/multi_step.json` |
| real RL training (D15/D17) | 19 committed GRPO updates (bounded off-policy lag-1), nonzero loss, real weight delta 10.4, guard ALLOW every step; success curve reported honestly (in-train reward, not held-out) | `rl_training/rl_training.json` |
| v0.2 variant matrix (F5-F8 ×3 variants) | 12/12 matched, normal 4/4 ALLOW, GATE PASS | `v0.2.0-dev/fault_matrix.json` |

## v0.2: fault families F5-F8 (formal, decision D12)

F5-F8 are FORMAL v0.2 families (D12, 2026-08-23) — the design doc §11
upgrade condition (frozen injection protocol + passing matrix) is met
(`docs/INJECTION_PROTOCOL_v02.md`). CPU-only; NOT part of the v0.1 matrix:

| Fault | Rule | Decision | Evidence |
|---|---|---|---|
| F5 split leakage | `D003_SPLIT_OVERLAP` (+ overlap report) | reject | `tests/frozen/f5_f8_v02/`, `artifacts/v0.2.0-dev/fault_matrix.json` |
| F6 evaluator alias | `R006_EVALUATOR_ALIAS` | quarantine | same |
| F7 event reorder | `L005_SCORING_AFTER_UPDATE` (+P007) | reject | same |
| F8 artifact mutation | `T001_ARTIFACT_HASH_MISMATCH` (isolated store) | reject | same |

`uv run grpo-guard v02-matrix --loop-dir artifacts/v0.1.0/loop` → 4/4 matched,
normal 4/4 allow, GATE PASS.

## v0.2.1: fault families F9-F10 (decision D16)

Task-agnostic families: **F9 reward injection** —
`R008_REWARD_VERIFIER_UNREGISTERED` (a reward event claiming a verifier
not in the registered evaluator registry, or with a wrong protocol hash →
reject); **F10 data poisoning** — `D004_PROMPT_CONTENT_MISMATCH` (the
prompt's token content substituted under the same id vs the frozen
`content_sha256s` registry → reject; this activates the manifest's
previously-unchecked content field).  Frozen fixtures 3/3 + normal 4/4,
GATE PASS (`configs/faults/f9_f10_v01.yaml`, `tests/frozen/f9_f10_v01/`).

## Docker (CPU demo stack)

`Dockerfile` + `docker-compose.yml` package the CPU-only stack: the
`verify` service attests the evidence chain
(`grpo-guard verify --artifact-dir /artifacts/v0.1.0 --events ...`) and
`panel` serves the Streamlit monitor on :8501 with the committed
artifacts mounted read-only.

```bash
docker compose build
docker compose run --rm verify     # checksums + event seals/order/refs
docker compose up panel            # http://localhost:8501
```

Honest note: no docker daemon was available in this project's
environment, so the images were NOT built/run here — the compose file
mirrors exactly the commands verified natively (`uv run grpo-guard
verify ...`, `streamlit run examples/monitor/panel.py ...`).

## Operations: monitor + alerts

`grpo-guard events --dir <events> [--type --component --code --prompt]`
searches the append-only event log; `grpo-guard alert-scan --dir <events>
--webhook <url>` POSTs every non-ALLOW decision to a webhook
(Slack-compatible or generic JSON).  `examples/monitor/panel.py` is a
Streamlit panel over the event log (decision distribution, reason codes,
lineage tracing, run health) for live demos.

## Task portability

The framework is not bound to Countdown: a second deterministic reward
adapter exists for GSM8K-style math QA (`src/grpo_guard/adapters/
gsm8k_reward.py` — extracts the last numeric value, compares to the
golden answer) with its own protocol hash, frozen samples and event
integration tests (`tests/test_gsm8k_reward.py`,
`tests/test_gsm8k_integration.py`).  The event schema, artifact store,
envelope and validator layers are untouched — the same pipeline binds any
deterministic rule set.

## Limitations (design doc §21)

- The v0.1 update consumed its own policy's trajectories (loss ≈ 0,
  ratio ≈ 1) — the gradient-impact evidence comes from the Day 4 paired
  replay, which runs at v1 weights + a documented deterministic drift.
- The canary is a behavior sketch (greedy tokens), not a per-byte proof.
- F5–F8 are formal v0.2 families (decision D12) but are not part of the
  v0.1 matrix; the v0.1 validator's general checks cover minimal F7/F8
  fixtures.
- No cryptographic tamper-resistance against a malicious producer
  (design doc §5.3).

## Repository layout

```
src/grpo_guard/       schema, store, validators, adapters, faults, replay, CLI
examples/countdown/   smoke + guarded closed-loop scripts
configs/              frozen workload / protocol / fault-matrix configs
tests/                unit, contract, property, frozen cases
artifacts/v0.1.0/     gate evidence (manifests, matrices, checksums)
DECISION_LOG.md       every deviation, recorded before first formal run
```

## License

Apache-2.0.
