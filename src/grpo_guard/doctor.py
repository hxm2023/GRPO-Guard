"""Environment self-check (``grpo-guard doctor``).

Diagnoses the runtime environment against the frozen compatibility
profile (``compatibility_profile.yaml``): installed package versions,
GPU availability, port conflicts and leftover vLLM server processes.
Exit 0 iff everything matches; every finding is printed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

GUARD_PORTS = list(range(8001, 8012)) + list(range(51216, 51226))


@dataclass
class DoctorReport:
    findings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        self.findings.append(f"OK   {msg}")

    def warn(self, msg: str) -> None:
        self.findings.append(f"WARN {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        self.findings.append(f"FAIL {msg}")


def _version(pkg: str) -> str | None:
    if pkg == "python":
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    try:
        import importlib.metadata

        return importlib.metadata.version(pkg)
    except Exception:
        return None


def _compare(actual: str | None, expected: str | None, name: str, report: DoctorReport) -> None:
    if expected is None:
        report.ok(f"{name}: not pinned in profile")
        return
    if actual is None:
        report.fail(f"{name}: not installed (profile expects {expected})")
        return
    if actual == expected:
        report.ok(f"{name}: {actual}")
    else:
        report.fail(f"{name}: installed {actual} != profile {expected}")


def run_doctor(profile_path: Path) -> DoctorReport:
    report = DoctorReport()
    profile = {}
    if profile_path.exists():
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    pinned = profile.get("pinned") or profile.get("versions") or {}

    for pkg, ver in (pinned or {}).items():
        _compare(_version(pkg), ver, pkg, report)
    if not pinned:
        # flat profile layout: top-level scalar keys that look like packages
        flat = {k: v for k, v in profile.items()
                if isinstance(v, str) and k in ("python", "torch", "transformers", "trl",
                                                "vllm", "accelerate", "cuda_runtime")}
        for pkg, ver in flat.items():
            if pkg == "cuda_runtime":
                continue
            _compare(_version(pkg), ver, pkg, report)
        if not flat:
            report.warn("profile has no recognizable version entries; skipping version checks")

    # GPU
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            for line in out.stdout.strip().splitlines():
                report.ok(f"GPU: {line.strip()}")
        except Exception as exc:
            report.warn(f"nvidia-smi failed: {exc}")
    else:
        report.warn("nvidia-smi not found (CPU-only environment)")

    # ports
    import socket

    busy = []
    for port in GUARD_PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                busy.append(port)
    if busy:
        report.warn(f"ports in use: {busy}")
    else:
        report.ok("guard ports 8001-8011/51216-51225 free")

    # leftover vLLM servers (any trl vllm-serve process)
    try:
        out = subprocess.run([sys.executable, "-c",
                              "import subprocess,sys;"
                              "r=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True);"
                              "print('\n'.join(l for l in r.stdout.splitlines() if 'vllm-serve' in l))"],
                             capture_output=True, text=True, timeout=30)
        lines = [l for l in out.stdout.strip().splitlines() if l]
        if lines:
            report.warn(f"{len(lines)} vllm-serve process(es) running")
        else:
            report.ok("no leftover vllm-serve processes")
    except Exception as exc:
        report.warn(f"process scan unavailable: {exc}")

    return report


def check_checkpoint(checkpoint_dir: Path) -> tuple[list[str], list[str]]:
    """Verify a committed checkpoint: PolicyManifest weights' sha256 vs disk.

    Returns (failures, warnings).  Failures = corrupted shards (hash
    mismatch on an existing file) or a missing manifest.  Warnings =
    missing shards — the repo intentionally stores light artifacts
    (design doc §18: full checkpoints are NOT published), so a fresh
    checkout legitimately lacks them; a FULL training environment must
    have zero warnings.
    """
    failures: list[str] = []
    warnings: list[str] = []
    manifest_path = checkpoint_dir / "policy_manifest.json"
    if not manifest_path.exists():
        return [f"{checkpoint_dir}: missing policy_manifest.json"], []
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = man.get("weights") or []
    if not weights:
        return [f"{checkpoint_dir}: manifest has no weights entries"], []
    import hashlib

    for w in weights:
        uri = w.get("uri", "")
        name = uri.rsplit("/", 1)[-1].removeprefix("artifact://")
        target = checkpoint_dir / name
        if not target.exists():
            warnings.append(f"{checkpoint_dir}/{name}: shard missing (light-artifact repo?)")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != w.get("sha256"):
            failures.append(f"{checkpoint_dir}/{name}: hash mismatch (corrupted)")
    return failures, warnings
