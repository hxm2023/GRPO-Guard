"""Canary stress check on autodl2 (decision D11): determinism under load.

One server load; 8 canary prompts (incl. near-max-context); 10 repeated
greedy sketches; every repetition must equal the baseline (drift 0) —
the canary suite is deterministic on a fixed weight set, so a nonzero
drift would indicate environment nondeterminism.

Output: <out>/canary_stress.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

MODEL_PATH = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/canary_stress_out"))
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8008"))
GROUP_PORT = int(os.environ.get("GRPO_GUARD_GROUP_PORT", "51223"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))


def log(msg: str) -> None:
    print(f"[canary-stress] {msg}", flush=True)


def main() -> int:
    from trl.generation.vllm_client import VLLMClient

    from examples.countdown.closed_loop import start_server, stop_server
    from grpo_guard.canary import CanarySuite, _token_diff

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 8 prompts incl. near-max-context
    prompts = [
        "2+2=",
        "Multiply 3, 7 and 11: ",
        "List the special token ids of this tokenizer: ",
        "Complete this long sentence with exactly five words: " + "word " * 900,
        "What is the capital of France? ",
        "Solve: 25*4-3 = ",
        "Translate 'good morning' to Chinese: ",
        "Write the next prime after 13: ",
    ]
    suite = CanarySuite(prompts=prompts, max_tokens=32)

    server = start_server(OUT_DIR / "vllm_server.log", port=VLLM_PORT, mem_util=0.35)
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{VLLM_PORT}", group_port=GROUP_PORT,
                            connection_timeout=300)

        def generate_fn(p, **kw):
            res = client.generate(p, n=1, temperature=0.0, top_p=1.0, top_k=1,
                                  max_tokens=kw.get("max_tokens", 32), logprobs=0)
            return res  # dict; CanarySuite handles dict returns

        baseline = suite.sketch(generate_fn)
        drifts = []
        for rep in range(10):
            sketch = suite.sketch(generate_fn)
            d = max(_token_diff(b, s) for b, s in zip(baseline, sketch))
            drifts.append(d)
        result = {
            "run_id": f"canary-stress-{int(time.time())}",
            "scope": "canary determinism under repeated load (8 prompts, 32 tokens, 10 repeats)",
            "prompts": len(prompts),
            "repeats": 10,
            "per_repeat_max_drift": drifts,
            "max_drift_any_repeat": max(drifts),
            "deterministic": all(d == 0 for d in drifts),
            "interpretation": "greedy canary sketches are deterministic on a fixed weight set "
                              "(drift 0 across 10 repeats) — any nonzero drift would indicate "
                              "environment nondeterminism",
        }
        (OUT_DIR / "canary_stress.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"deterministic={result['deterministic']} max_drift={max(drifts)}")
        log("CANARY STRESS DONE")
        return 0 if result["deterministic"] else 1
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
