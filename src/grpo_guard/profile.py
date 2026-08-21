"""Compatibility profile freeze (design doc §4.1.1).

The profile pins the exact observed versions on the GPU box BEFORE any
adapter work: python/torch/transformers/trl/vllm/accelerate/cuda runtime and
driver, the model id + immutable revision, GPU layout and the serve command.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml


def _server_env(server: str) -> dict:
    """Read the pinned versions from the GPU box via ssh."""
    script = (
        "cd /root/autodl-tmp/grpo-guard && source .venv/bin/activate && "
        "python - <<'PY'\n"
        "import json, torch, transformers, trl, vllm, accelerate, platform, subprocess\n"
        "out = {\n"
        "  'python': platform.python_version(),\n"
        "  'torch': torch.__version__,\n"
        "  'transformers': transformers.__version__,\n"
        "  'trl': trl.__version__,\n"
        "  'vllm': vllm.__version__,\n"
        "  'accelerate': accelerate.__version__,\n"
        "}\n"
        "print(json.dumps(out))\n"
        "PY"
    )
    res = subprocess.run(
        ["ssh", server, script], capture_output=True, text=True, timeout=120
    )
    if res.returncode != 0:
        raise RuntimeError(f"ssh env probe failed: {res.stderr[-500:]}")
    env = json.loads(res.stdout.strip().splitlines()[-1])

    probe = (
        "nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader; "
        "nvidia-smi --query-gpu=index,memory.total --format=csv,noheader"
    )
    res = subprocess.run(["ssh", server, probe], capture_output=True, text=True, timeout=60)
    gpus = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append({"index": parts[0], "name": parts[1], "driver": parts[2] if len(parts) > 2 else ""})
    env["cuda_driver"] = gpus[0].get("driver", "")
    return env


def freeze_compatibility_profile(out_path: Path, server: str = "autodl2") -> Path:
    env = _server_env(server)
    profile = {
        "profile_id": f"cuda-{env['torch'].split('+')[1]}-server-v01",
        "python": env["python"],
        "torch": env["torch"],
        "cuda_runtime": "12.8",
        "cuda_driver": env.get("cuda_driver", ""),
        "transformers": env["transformers"],
        "trl": env["trl"],
        "vllm": env["vllm"],
        "accelerate": env["accelerate"],
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "TBD-after-first-snapshot",
        "trl_mode": "server",
        "trainer_cuda_visible_devices": [0],
        "rollout_cuda_visible_devices": [1],
        "serve_command": ["trl", "vllm-serve", "--model", "Qwen/Qwen3-4B", "--port", "8000"],
        "trainer_config": {"use_vllm": True, "vllm_mode": "server"},
        "upstream_sync_adapter": {
            "python_qualname": "trl.vllm.vllm_client.VLLMClient.update_named_param",
            "source_file_sha256": "",
            "request_or_collective": "http-single-request",
        },
        "official_smoke_passed": False,
    }
    out_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    return out_path
