"""P0-2 experiment: detect a SILENT NO-OP sync (the original static-rollout
accident) via server-vs-trainer greedy sketch comparison.

The original incident: update_named_param silently did nothing; the
runtime kept serving the OLD policy while metadata advanced.  A
v0-baseline canary cannot see this during training (weights legitimately
move — D17), so the guard must compare the SERVER's greedy sketch against
the TRAINER's own greedy decode of the CURRENT weights: if they diverge,
the runtime is not actually serving the committed policy — reject before
the next rollout is consumed.

Design (GPU run needed; CPU unit test covers the decision logic):

1. calibrate the canary suite on v0 (5 reloads, frozen tolerance);
2. sync policy v1 to the server via update_named_param — either REAL or a
   success-returning NO-OP stub;
3. sketch the server (greedy, temperature 0);
4. greedy-decode the same prompts with the TRAINER's current weights;
5. drift > tolerance  => runtime stale => reject consumption (P008 path);
   drift <= tolerance => runtime actually loaded the new weights.

The stub's API is identical (returns success) — only the data-plane
sketch comparison exposes it.  This is the exact "loss curves look fine
but the policy is static" scenario, made machine-checkable.

Outputs: <out>/sync_noop_result.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/sync_noop_out"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))


class NoopSyncStub:
    """update_named_param that RETURNS SUCCESS but changes nothing."""

    def __init__(self):
        self.calls = 0

    def update_named_param(self, name, param) -> None:
        self.calls += 1  # silently drop the weight


def greedy_decode(model, tokenizer, prompts: list[str], max_tokens: int = 8) -> list[list[int]]:
    """Deterministic greedy decode with the GIVEN model weights (no server)."""
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
    """Max token drift between the server's sketch and the trainer's own."""
    from grpo_guard.canary import _token_diff

    return max(_token_diff(a, b) for a, b in zip(server_sketch, model_sketch))


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.generation.vllm_client import VLLMClient

    from examples.countdown.closed_loop import start_server, stop_server, _unpack_gen
    from grpo_guard.canary import CanarySuite

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # D18: trainer + vLLM BOTH on GPU1 to share the card with agent-ttrl
    # (GPU0 is occupied by its full-finetune; mem_util 0.3 keeps us inside
    # the remaining ~50GB of GPU1).
    # CUDA_VISIBLE_DEVICES=1 (launch script) remaps physical GPU1 to cuda:0
    model = AutoModelForCausalLM.from_pretrained(
        os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B"),
        torch_dtype=torch.bfloat16, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B"),
        trust_remote_code=True)

    suite = CanarySuite()
    server = start_server(OUT_DIR / "vllm_server.log", mem_util=0.3, device="1")
    try:
        client = VLLMClient(base_url=f"http://127.0.0.1:{8009}", group_port=51224,
                            connection_timeout=300)

        def server_sketch():
            return suite.sketch(lambda p, **kw: _unpack_gen(client.generate(
                p, n=1, temperature=0.0, top_p=1.0, top_k=1, max_tokens=8, logprobs=0)))

        baseline = server_sketch()  # server = v0

        # REAL sync
        client.init_communicator(device=torch.device("cuda:0"))
        real_calls = 0
        for name, param in model.named_parameters():
            client.update_named_param(name, param.data)
            real_calls += 1
        real_server = server_sketch()
        real_model = greedy_decode(model, tokenizer, suite.prompts)
        real_drift = compare_sketches(real_server, real_model)
        real_result = {"sync": "real", "calls": real_calls,
                       "server_vs_model_drift": real_drift,
                       "runtime_loaded": real_drift <= 0}

        # NO-OP sync (returns success, drops weights) — same weights as above
        # would be pushed, but the stub never applies them.
        stub = NoopSyncStub()
        for name, param in model.named_parameters():
            stub.update_named_param(name, param.data)
        noop_server = server_sketch()  # server unchanged (still v0 weights)
        noop_model = greedy_decode(model, tokenizer, suite.prompts)
        noop_drift = compare_sketches(noop_server, noop_model)
        noop_result = {"sync": "noop", "calls": stub.calls,
                       "server_vs_model_drift": noop_drift,
                       "runtime_loaded": noop_drift <= 0,
                       "detected": noop_drift > 0}

        result = {
            "scope": "P0-2: silent no-op sync detection via server-vs-trainer "
                     "greedy sketch (the original static-rollout accident)",
            "real": real_result,
            "noop": noop_result,
            "verdict": ("runtime staleness DETECTED" if noop_result["detected"]
                        else "no-op NOT detected (tolerance too loose?)"),
        }
        (OUT_DIR / "sync_noop_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main())
