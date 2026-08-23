"""doctor + checkpoint verification tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from grpo_guard.doctor import check_checkpoint, run_doctor


def test_check_checkpoint_ok(tmp_path: Path):
    ckpt = tmp_path / "ckpt_v1"
    ckpt.mkdir()
    blob = b"fake-tensor-bytes"
    (ckpt / "model-00001-of-00005.safetensors").write_bytes(blob)
    (ckpt / "policy_manifest.json").write_text(json.dumps({
        "weights": [{"uri": "artifact://model-00001-of-00005.safetensors",
                     "sha256": hashlib.sha256(blob).hexdigest()}],
    }), encoding="utf-8")
    failures, warnings = check_checkpoint(ckpt)
    assert failures == [] and warnings == []


def test_check_checkpoint_hash_mismatch(tmp_path: Path):
    ckpt = tmp_path / "ckpt_v1"
    ckpt.mkdir()
    (ckpt / "model-00001-of-00005.safetensors").write_bytes(b"real-bytes")
    (ckpt / "policy_manifest.json").write_text(json.dumps({
        "weights": [{"uri": "artifact://model-00001-of-00005.safetensors", "sha256": "0" * 64}],
    }), encoding="utf-8")
    failures, _ = check_checkpoint(ckpt)
    assert any("hash mismatch" in f for f in failures)


def test_check_checkpoint_missing_shard_is_warning(tmp_path: Path):
    ckpt = tmp_path / "ckpt_v1"
    ckpt.mkdir()
    (ckpt / "policy_manifest.json").write_text(json.dumps({
        "weights": [{"uri": "artifact://model-00002-of-00005.safetensors", "sha256": "0" * 64}],
    }), encoding="utf-8")
    failures, warnings = check_checkpoint(ckpt)
    assert failures == []
    assert any("shard missing" in w for w in warnings)


def test_doctor_missing_profile_does_not_crash(tmp_path: Path):
    report = run_doctor(tmp_path / "nonexistent.yaml")
    assert report.findings  # runs without crashing, findings recorded
