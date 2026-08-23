"""Evidence-chain verification (production ops tool).

``grpo-guard verify`` checks an artifact directory end-to-end:

1. **Checksums** — every file listed in ``SHA256SUMS`` hashes correctly
   (no tampered/missing evidence).
2. **Event seals** — every event's ``event_sha256`` is self-consistent
   (canonical JSON over the payload excluding the hash itself).
3. **Append-only order** — ``lifecycle_seq`` strictly increases across the
   event log (no reordering / reuse).
4. **Reference integrity** — every EventRef (input_events,
   source_generation_event, etc.) points at an existing event whose
   ``event_sha256`` matches (no dangling or swapped references).

Failures are reported with exact paths; exit code 0 iff everything
passes.  This is the same check a production operator would run
periodically to attest the evidence chain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VerifyReport:
    ok: bool = True
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def fail(self, what: str) -> None:
        self.ok = False
        self.failures.append(what)

    def record(self, what: str) -> None:
        self.checks.append(what)


def verify_checksums(artifact_dir: Path) -> list[str]:
    """Check SHA256SUMS over the artifact dir. Returns failure lines."""
    failures = []
    sums_file = artifact_dir / "SHA256SUMS"
    if not sums_file.exists():
        return [f"{sums_file}: missing SHA256SUMS"]
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            expected, rel = line.split("  ", 1)
        except ValueError:
            failures.append(f"{sums_file}: malformed line {line[:40]}")
            continue
        target = artifact_dir / rel
        if not target.exists():
            failures.append(f"missing: {rel}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"hash mismatch: {rel}")
    return failures


def _ref_event_ids(payload: dict) -> list[str]:
    """Collect every event_id referenced by an event payload."""
    ids: list[str] = []
    for key in ("input_events", "source_generation_event", "sync_event",
                "authoritative_behavior_logprob_event", "parent_identity_decision",
                "reward_event", "preupdate_envelope", "preupdate_validation_decision",
                "update_input_event"):
        ref = payload.get(key)
        if isinstance(ref, list):
            for r in ref:
                if isinstance(r, dict) and r.get("event_id"):
                    ids.append(r["event_id"])
        elif isinstance(ref, dict) and ref.get("event_id"):
            ids.append(ref["event_id"])
    return ids


def verify_events(events_dir: Path) -> list[str]:
    """Verify seal, ordering and reference integrity of an event log."""
    failures: list[str] = []
    by_id: dict[str, tuple[dict, str]] = {}  # event_id -> (payload, file)
    seq_seen: list[int] = []

    for p in sorted(events_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            failures.append(f"{p.name}: unreadable ({exc})")
            continue
        eid = payload.get("event_id", "?")
        by_id[eid] = (payload, p.name)
        # 2. seal self-consistency
        expected = payload.get("event_sha256")
        if not expected:
            failures.append(f"{p.name}: missing event_sha256")
        else:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False)
            # canonical re-serialization must exclude event_sha256 and match
            from grpo_guard.store.canonical_json import canonical_dumps

            payload_no_sha = {k: v for k, v in payload.items() if k != "event_sha256"}
            recalc = hashlib.sha256(canonical_dumps(payload_no_sha)).hexdigest()
            if recalc != expected:
                failures.append(f"{p.name}: seal mismatch (event_sha256 not self-consistent)")
        # 3. lifecycle ordering
        seq = payload.get("lifecycle_seq")
        if seq is not None:
            seq_seen.append((seq, p.name))
    # strictly increasing: sort by seq first (file names are NOT chronological)
    seq_seen.sort()
    for (s1, f1), (s2, f2) in zip(seq_seen, seq_seen[1:]):
        if s2 <= s1:
            failures.append(f"{f2}: lifecycle_seq {s2} not > previous {s1} ({f1})")

    # 3b. update_committed lineage: parent + 1 == output (P0-3)
    for eid, (payload, fname) in by_id.items():
        if payload.get("event_type") == "update_committed":
            parent = payload.get("parent_policy_version")
            output = payload.get("output_policy_version")
            if parent is not None and output is not None and output != parent + 1:
                failures.append(
                    f"{fname}: update_committed parent={parent} but output={output} "
                    "(parent+1 == output violated)")

    # 4. reference integrity
    for eid, (payload, fname) in by_id.items():
        for ref_id in _ref_event_ids(payload):
            ref_payload = by_id.get(ref_id)
            if ref_payload is None:
                failures.append(f"{fname}: dangling reference to event {ref_id}")
                continue
            # if the reference carries a sha, it must match the target
            sha = None
            for key in ("input_events", "source_generation_event", "sync_event",
                        "authoritative_behavior_logprob_event", "parent_identity_decision",
                        "reward_event"):
                ref = payload.get(key)
                items = ref if isinstance(ref, list) else [ref]
                for r in items:
                    if isinstance(r, dict) and r.get("event_id") == ref_id and r.get("event_sha256"):
                        sha = r["event_sha256"]
            if sha and ref_payload[0].get("event_sha256") != sha:
                failures.append(f"{fname}: reference to {ref_id} has wrong event_sha256")
    return failures


def verify_artifact_dir(artifact_dir: Path, events_dir: Path | None = None) -> VerifyReport:
    report = VerifyReport()
    sum_failures = verify_checksums(artifact_dir)
    for f in sum_failures:
        report.fail(f)
    report.record(f"checksums: {'ok' if not sum_failures else f'{len(sum_failures)} failures'}")

    if events_dir is not None:
        ev_failures = verify_events(events_dir)
        for f in ev_failures:
            report.fail(f)
        report.record(f"events: {'ok' if not ev_failures else f'{len(ev_failures)} failures'}")
    return report
