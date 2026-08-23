"""P0-2 experiment (D18 simplified): detect a STALE runtime — server still
serving v0 while the trainer holds the trained v20 — via server-vs-trainer
greedy sketch comparison.

This is the original static-rollout accident made machine-checkable: the
runtime keeps serving an OLD policy while the committed weights advanced.
No weight-sync is performed (no communicator needed): the trainer loads
the trained checkpoint (ckpt_v20, ||dθ||≈9.5 from the full D18 RL run)
and greedy-decodes the canary prompts locally; the server still serves
v0.  If the sketches diverge, the runtime is provably stale → the guard
would reject consuming its rollouts.

Outputs: <out>/sync_noop_result.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/sync_noop_out"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
CKPT_DIR = Path(os.environ.get("GRPO_GUARD_CKPT", "/root/autodl-tmp/grpo-guard/rl_out/ckpt_v20"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))


def greedy_decode(model, tokenizer, prompts: list[str], max_tokens: int = 8) -> list[list[int]]:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for p in prompts:
            ids = tokenizer(p, return_tensors="pt")["input_ids"].to(next(model.parameters()).device)
            gen = model.generate(ids, do_sample=False, max_new_tokens=max_tokens)
            out.append(gen[0, ids.shape[1]:].tolist())
    model.train()
    return out


def compare_sketches(server_sketch, model_sketch) -> int:
    from grpo_guard.canary import _token_diff

    return max(_token_diff(a, b) for a, b in zip(server_sketch, model_sketch))


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    from examples.countdown.closed_loop import start_server, stop_server, _unpack_gen
    from grpo_guard.canary import CanarySuite
    from safetensors.torch import load_file as st_load

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B"),
        torch_dtype=torch.bfloat16, device_map="cuda:0")
    # load the TRAINED checkpoint (weights really moved vs v0)
    for shard in sorted(CKPT_DIR.glob("model-*.safetensors")):
        tensors = st_load(str(shard))
        model.load_state_dict(tensors, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B"),
        trust_remote_code=True)

    suite = CanarySuite()
    server = start_server(OUT_DIR / "vllm_server.log", port=8009, mem_util=0.3, device="1")
    try:
        client = VLLMClient(base_url="http://127.0.0.1:8009", group_port=51224,
                            connection_timeout=300)
        server_sketch = suite.sketch(lambda p, **kw: _unpack_gen(client.generate(
            p, n=1, temperature=0.0, top_p=1.0, top_k=1, max_tokens=8, logprobs=0)))
        model_sketch = greedy_decode(model, tokenizer, suite.prompts)
        drift = compare_sketches(server_sketch, model_sketch)

        result = {
            "scope": "P0-2 stale-runtime detection (static-rollout accident): server serves v0, "
                     "trainer holds trained v20 (||dθ||≈9.5); server-vs-trainer greedy sketch",
            "checkpoint_loaded": str(CKPT_DIR),
            "server_vs_model_drift": drift,
            "runtime_stale_detected": drift > 0,
            "verdict": ("STALE RUNTIME DETECTED — guard would reject consuming its rollouts"
                        if drift > 0 else "runtime matches trainer (no staleness)"),
        }
        (OUT_DIR / "sync_noop_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
