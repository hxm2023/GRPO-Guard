"""Evidence-chain verification tool tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from grpo_guard import testing
from grpo_guard.verify import verify_artifact_dir, verify_checksums, verify_events


def _write_events(tmp_path: Path) -> Path:
    t = testing.build_trajectory()
    d = tmp_path / "events"
    d.mkdir()
    for eid, ev in t.events.items():
        payload = ev.model_dump(mode="json")
        (d / f"{eid}.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def test_verify_ok_on_valid_evidence(tmp_path: Path):
    events = _write_events(tmp_path)
    # build a SHA256SUMS over some artifact files
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    f = artifact_dir / "report.txt"
    f.write_text("evidence", encoding="utf-8")
    (artifact_dir / "SHA256SUMS").write_text(
        f"{hashlib.sha256(b'evidence').hexdigest()}  report.txt\n", encoding="utf-8")
    report = verify_artifact_dir(artifact_dir, events)
    assert report.ok, report.failures


def test_verify_detects_tampered_checksum(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    f = artifact_dir / "report.txt"
    f.write_text("evidence", encoding="utf-8")
    (artifact_dir / "SHA256SUMS").write_text(
        f"{'0' * 64}  report.txt\n", encoding="utf-8")
    report = verify_artifact_dir(artifact_dir)
    assert not report.ok
    assert any("hash mismatch" in x for x in report.failures)


def test_verify_detects_seal_break(tmp_path: Path):
    events = _write_events(tmp_path)
    # tamper with one event without updating its seal
    victim = sorted(events.glob("*.json"))[0]
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["component_id"] = "tampered"
    victim.write_text(json.dumps(payload), encoding="utf-8")
    failures = verify_events(events)
    assert any("seal mismatch" in x for x in failures)


def test_verify_detects_dangling_reference(tmp_path: Path):
    events = _write_events(tmp_path)
    victim = sorted(events.glob("*.json"))[0]
    payload = json.loads(victim.read_text(encoding="utf-8"))
    payload["input_events"] = [{"event_id": "nonexistent-event", "event_sha256": "0" * 64}]
    victim.write_text(json.dumps(payload), encoding="utf-8")
    failures = verify_events(events)
    assert any("dangling reference" in x for x in failures)
