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


def build_report(artifact_dir: Path, commit: str = "", update: bool = False) -> dict:
    """Build the release bundle: run_manifest.json + REPORT.md + SHA256SUMS.

    Default is no-overwrite (fresh dirs only).  ``update=True`` merges a new
    manifest entry and REGENERATES SHA256SUMS over the current artifact set —
    the explicit path for release flows where artifacts changed after the
    initial bundle (canonical outputs are never silently clobbered).
    """
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_manifest.json"
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    if manifest_path.exists() and not update:
        raise FileExistsError(f"{manifest_path} exists (no-overwrite canonical output; pass --update to refresh)")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("updates", []).append({
            "updated_at_utc": now,
            "commit": commit or "uncommitted",
            "platform": {"system": platform.system(), "python": platform.python_version()},
        })
    else:
        manifest = {
            "release": "v0.1.0",
            "created_at_utc": now,
            "commit": commit,
            "platform": {"system": platform.system(), "python": platform.python_version()},
            "stages": {},
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    report_md = out / "REPORT.md"
    if not report_md.exists():
        report_md.write_text(
            "# GRPO-Guard v0.1.0 — run report\n\n"
            f"- created: {manifest['created_at_utc']}\n"
            f"- commit: {commit or 'uncommitted'}\n\n"
            "Filled in per-day as gates pass (design doc §16).\n",
            encoding="utf-8",
        )

    sums = out / "SHA256SUMS"
    sums.write_text("\n".join(_sha256sums_lines(out)) + "\n", encoding="utf-8")
    return {"run_manifest": str(manifest_path), "report_md": str(report_md), "sha256sums": str(sums)}
