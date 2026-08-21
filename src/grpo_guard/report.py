"""Release report builder: run_manifest.json + REPORT.md + SHA256SUMS
(design doc §13, §19.2; no-overwrite canonical outputs)."""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256sums_lines(root: Path) -> list[str]:
    return sorted(f"{_sha256(p)}  {p.relative_to(root).as_posix()}" for p in root.rglob("*") if p.is_file())


def build_report(artifact_dir: Path, commit: str = "") -> dict:
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists (no-overwrite canonical output)")

    manifest = {
        "release": "v0.1.0",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "commit": commit,
        "platform": {"system": platform.system(), "python": platform.python_version()},
        "stages": {},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    report_md = out / "REPORT.md"
    if report_md.exists():
        raise FileExistsError(f"{report_md} exists (no-overwrite canonical output)")
    report_md.write_text(
        "# GRPO-Guard v0.1.0 — run report\n\n"
        f"- created: {manifest['created_at_utc']}\n"
        f"- commit: {commit or 'uncommitted'}\n\n"
        "Filled in per-day as gates pass (design doc §16).\n",
        encoding="utf-8",
    )

    sums = out / "SHA256SUMS"
    if sums.exists():
        raise FileExistsError(f"{sums} exists (no-overwrite canonical output)")
    sums.write_text("\n".join(_sha256sums_lines(out)) + "\n", encoding="utf-8")
    return {"run_manifest": str(manifest_path), "report_md": str(report_md), "sha256sums": str(sums)}
