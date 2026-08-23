# Postmortems — real incidents from the GRPO-Guard project

Three incidents that actually happened during this project's development.
Each is written in the standard postmortem shape (impact → root cause →
detection → fix → prevention).  All timestamps are 2026-08-23, all links
trace to commits/artifacts in this repo.

## PM-1: Canary fail-closed aborts legitimate RL training (drift 8)

**Impact**: two RL-training runs aborted after step 1 — the guard
correctly rejected, but the *policy* was wrong for the context: in
training the weights are *supposed* to move.  ~2.5 GPU·h and two
calibration cycles (5 server reloads each) burned.

**Root cause**: `CanarySuite.check` compares the current greedy sketch
against the frozen v0 baseline with calibration tolerance 0.  A single
off-policy GRPO update (lr=1e-3, then lr=1e-4) shifted greedy tokens by
7-8 — the canary fired.  The canary's "frozen-weight consistency"
semantics are correct for deployment/reload/sync-fidelity checks, not
for legitimate policy updates.

**Detection**: `RuntimeError: canary MISMATCH after v1 sync:
{'max_token_drift': 8}` — the fail-closed path worked exactly as
designed (P008 semantics).

**Fix**: decision D17 — during training the canary becomes a **drift
monitor** (recorded per step, `canary_drift` in every step record), while
P008 fail-closed stays active for non-training checks (loop syncs, the
canary-mismatch experiment).  Committed as `ad059d5`.

**Prevention**: document the two canary modes (gate vs monitor) in
`docs/UPSTREAM_FEEDBACK.md` context and the script docstring.

## PM-2: vLLM engine dies mid-training; evidence recovered from the event log

**Impact**: step 20 of the real RL training loop hung forever — the
server process stayed alive but released all GPU memory (engine core
died silently).  The run could not continue; budget was exhausted so a
rerun was impossible.

**Root cause**: `trl vllm-serve` engine core died during step 20's
rollout.  Logs show `ImportError: libnvrtc.so.13` in the shutdown path
of an unrelated earlier server (environment drifted — another project
re-resolved packages on the shared box); the engine failure mode left
the parent process hanging instead of failing fast.

**Detection**: the operator loop noticed the log tail stuck at
"step 19 done" with GPU at 0 MiB while processes remained alive.
The event log (3304 append-only events) contained the full 19-step
record.

**Fix**: `examples/countdown/recover_rl.py` rebuilt `rl_training.json`
from the event log + run log — every number traced to its source,
`recovered: true` marked in the json.  No fabrication: loss/ratio/weight
delta came from the run log lines, success rates from reward events
grouped by request_id.

**Prevention**: (1) the verify tool (PM-3 companion) can attest an event
log end-to-end; (2) future runs should persist per-step metrics to the
event stream itself (e.g. a `training_step` event) so recovery never
needs the run log.

## PM-3: Evidence-chain integrity vs. CI checkout (SHA256SUMS drift)

**Impact**: the new CI checksum gate failed on the first two pushes —
`sha256sum -c` over `artifacts/v0.1.0` failed for every file, and then
for the smoke logs.  The evidence chain looked broken when it was not.

**Root cause**: two compounding issues. (1) Windows working tree had
CRLF line endings while CI checked out LF — hashes computed locally did
not match the committed bytes. (2) `smoke/logs/*.log` files were on
disk but gitignored — `SHA256SUMS` covered files that a fresh checkout
does not contain.

**Detection**: CI step `Evidence chain integrity (SHA256SUMS)` failed;
local `sha256sum -c` passed (local CRLF files), CI failed (LF files) —
the discrepancy *was* the signal.

**Fix**: `.gitattributes` pins `artifacts/**` to LF; `SHA256SUMS` now
covers only committed files (verified against `git show :path` bytes);
the integrity gate runs in CI on every push.

**Prevention**: this is exactly what the gate is for — the checksum
integrity step now runs before any release claim.

---

Each postmortem's numbers trace to commits and artifacts in this repo;
the honest-reporting discipline (D-notes in DECISION_LOG.md) is what
made these recoverable.
