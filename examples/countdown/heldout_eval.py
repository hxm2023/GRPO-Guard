"""P1-2 held-out evaluation (D18): frozen held-out GSM8K-style prompts.

Evaluates the SAME frozen 8 prompts (disjoint from the 8 training
prompts) with greedy decode under (a) the base v0 weights and (b) the
trained ckpt_v10 weights from the held-out training run — reporting
held-out success rates.  This is the honest held-out evidence the P1-2
3-seed study lacked (its curves were in-train rollout rewards).

Outputs: <out>/heldout_eval.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/heldout_out"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
CKPT_DIR = Path(os.environ.get("GRPO_GUARD_CKPT", "/root/autodl-tmp/grpo-guard/heldout_train/ckpt_v10"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

HELDOUT = [
    ("A farmer has 12 cows and buys 7 more. How many cows does he have?", 19),
    ("Sara has 4 bags with 3 apples in each. How many apples in total?", 12),
    ("A bus starts with 20 passengers. 5 get off and 8 get on. How many are on the bus?", 23),
    ("A rectangle is 6 units wide and 4 units tall. What is its area?", 24),
    ("Tom reads 5 pages a day for a whole week. How many pages does he read?", 35),
    ("There are 3 boxes with 6 pens each, and 4 extra pens. How many pens total?", 22),
    ("A shop sells 12 cupcakes on Monday and twice that on Tuesday. How many on Tuesday?", 24),
    ("A train covers 45 miles in 1 hour. How far does it go in 4 hours?", 180),
]


def evaluate(model, tokenizer) -> dict:
    import torch

    from grpo_guard.adapters.gsm8k_reward import gsm8k_rule_verifier

    model.eval()
    correct = 0
    per_q = []
    with torch.no_grad():
        for q, ans in HELDOUT:
            ids = tokenizer(f"{q}\nAnswer with only the final number.", return_tensors="pt")["input_ids"]
            gen = model.generate(ids.to(model.device), do_sample=False, max_new_tokens=16)
            text = tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            comps = gsm8k_rule_verifier(text, float(ans))
            correct += 1 if comps["correctness"] == 1.0 else 0
            per_q.append({"question": q[:40], "golden": ans, "completion": text.strip()[:40],
                          "correct": comps["correctness"] == 1.0})
    model.train()
    return {"n": len(HELDOUT), "correct": correct,
            "accuracy": correct / len(HELDOUT), "per_question": per_q}


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file as st_load

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # (a) base v0
    base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                device_map="cuda:0")
    v0 = evaluate(base, tokenizer)

    # (b) trained ckpt_v10
    trained = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                   device_map="cuda:0")
    for shard in sorted(CKPT_DIR.glob("model-*.safetensors")):
        trained.load_state_dict(st_load(str(shard)), strict=False)
    v10 = evaluate(trained, tokenizer)

    result = {
        "scope": "P1-2 held-out evaluation (D18): frozen 8 prompts disjoint from "
                 "training, greedy decode, base v0 vs trained v10",
        "held_out_prompts": len(HELDOUT),
        "base_v0": v0,
        "trained_v10": v10,
        "held_out_delta": round(v10["accuracy"] - v0["accuracy"], 4),
    }
    (OUT_DIR / "heldout_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
