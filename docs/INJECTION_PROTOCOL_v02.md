# v0.2 Injection Protocol (F5-F8) — frozen 2026-08-22

Design doc §11: F7/F8 (and F5/F6) become **formal fault families** only
after v0.2 freezes the complete injection protocol.  This document is that
freeze.  It binds the injectors, the pre-registered expectations
(`configs/faults/f5_f8_v02.yaml`), the frozen fixtures
(`tests/frozen/f5_f8_v02/`), and the matrix runners (`matrix_v02.py`,
`examples/countdown/online_matrix.py`) to the reason-code rules.

## Protocol invariants (identical spirit to F1-F4)

1. **One fault per case**: an injector mutates exactly one target field of a
   frozen producer artifact; nothing else changes.
2. **Pre-registered expectations**: every variant's expected decision and
   required reason codes are fixed in `configs/faults/f5_f8_v02.yaml`
   BEFORE the matrix runs; no post-hoc edits (honesty rules).
3. **Real artifacts**: the matrix runs against the committed closed-loop
   GenerationEvents (and, online, against REAL server rollouts).
4. **Quarantine/reject never consumed**: enforced by the guarded update's
   ALLOW verifier; F5-F8 decisions follow the same consumption ban.
5. **Isolated mutation**: F8 mutates a CLONE of the artifact store — the
   shared evidence is never touched.

## Family definitions

| ID | Fault | Injection (canonical) | Rule | Decision |
|---|---|---|---|---|
| F5 | split leakage | trajectory's prompt listed in a SECOND split manifest | `D003_SPLIT_OVERLAP` | reject |
| F6 | evaluator alias | train reward event produced under the DECLARED eval protocol | `R006_EVALUATOR_ALIAS` | quarantine |
| F7 | event reorder | scoring event placed after the consuming update | `L005_SCORING_AFTER_UPDATE` (+P007) | reject |
| F8 | artifact mutation | a sealed event's blob overwritten after sealing | `T001_ARTIFACT_HASH_MISMATCH` | reject |

## Variant matrix (pre-registered)

| family | canonical | boundary | held-out |
|---|---|---|---|
| F5 | leak into held_out | leak into held_out + calibration (triple) | leak into calibration |
| F6 | protocol collision | no alias → allow (negative control) | collision + eval-style reward_version |
| F7 | scoring late (L005) | scoring late + different scorer policy (L003+L005) | sync late (P007) |
| F8 | sequence blob | loss-mask blob | logprobs blob |

Run: `uv run grpo-guard v02-matrix --loop-dir artifacts/v0.1.0/loop`

## Verification status (2026-08-22)

- Offline (committed loop artifacts): **12/12 matched**, normal 4/4 ALLOW,
  GATE PASS.
- Online (autodl2, real server rollouts, `online_matrix.py`): F5-F8 4/4
  (canonical), F1-F4 4/4 + normal 4/4.
- Frozen fixtures: 16 cases, contract-check PASS, no-overwrite.
- Full suite: green.

## Upgrade path

v0.2 release = this protocol + the online matrix results + the v0.2 docs.
Anything not listed here (e.g. F9+) requires a NEW protocol version
(`INJECTION_PROTOCOL_v03.md`) and must not be bolted onto v0.2.
