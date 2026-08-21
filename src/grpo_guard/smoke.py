"""GPU smoke orchestrator: sync repo → run official TRL+vLLM server-mode
smoke on the GPU box → collect result (design doc §4.1.1 Compatibility Gate).
"""

from __future__ import annotations

import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

REPO_DIR_SERVER = "/root/autodl-tmp/grpo-guard/repo"
SMOKE_OUT_SERVER = "/root/autodl-tmp/grpo-guard/smoke_out"


def _run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _sync_repo(server: str, repo_root: Path) -> None:
    """Upload the repo (minus venv/git) to the server."""
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tar_path = Path(tmp.name)
    excludes = (".venv", ".git", ".pytest_cache", "__pycache__", ".hypothesis", "artifacts", "scratch")
    with tarfile.open(tar_path, "w:gz") as tar:
        for child in sorted(repo_root.iterdir()):
            if child.name in excludes or child.name.startswith("."):
                continue
            tar.add(child, arcname=child.name)
    _run(["scp", str(tar_path), f"{server}:/tmp/grpo-guard-repo.tar.gz"], timeout=600)
    tar_path.unlink()
    script = f"rm -rf {REPO_DIR_SERVER} && mkdir -p {REPO_DIR_SERVER} && tar -xzf /tmp/grpo-guard-repo.tar.gz -C {REPO_DIR_SERVER}"
    res = _run(["ssh", server, script], timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"repo sync failed: {res.stderr[-500:]}")


def run_smoke(config_path: Path, out_dir: Path, server: str = "autodl2") -> dict:
    _sync_repo(server, config_path.parent.parent)  # project root
    script = (
        f"cd {REPO_DIR_SERVER} && bash scripts/run_gpu_smoke.sh {REPO_DIR_SERVER} {SMOKE_OUT_SERVER}"
    )
    res = _run(["ssh", server, f"nohup bash -c '{script}' > {SMOKE_OUT_SERVER}/smoke_runner.log 2>&1 & echo LAUNCHED"], timeout=60)
    if "LAUNCHED" not in res.stdout:
        raise RuntimeError(f"smoke launch failed: {res.stdout} {res.stderr}")
    # poll
    import time

    deadline = time.time() + 45 * 60
    while time.time() < deadline:
        time.sleep(20)
        r = _run(["ssh", server, f"tail -3 {SMOKE_OUT_SERVER}/smoke_runner.log 2>/dev/null; test -f {SMOKE_OUT_SERVER}/smoke_result.json && echo RESULT_READY"], timeout=60)
        if "RESULT_READY" in r.stdout:
            break
    if "RESULT_READY" not in r.stdout:
        tail = _run(["ssh", server, f"tail -20 {SMOKE_OUT_SERVER}/smoke_runner.log"], timeout=60)
        raise RuntimeError(f"smoke did not finish in 45min; tail:\n{tail.stdout[-2000:]}")

    result = json.loads(_run(["ssh", server, f"cat {SMOKE_OUT_SERVER}/smoke_result.json"], timeout=60).stdout)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
