"""Evaluate a TRL-trained checkpoint on the 16 frozen held-out prompts.

Runs in a SEPARATE process (loads only ~8GB) so the training process can
exit right after the checkpoint save — the evaluation cannot be killed by
GPU contention mid-run.

Env: GRPO_GUARD_CKPT (trained checkpoint dir), GRPO_GUARD_MODEL_PATH,
GRPO_GUARD_OUT.  Outputs: <out>/eval_heldout.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MODEL_ID = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
CKPT = os.environ.get("GRPO_GUARD_CKPT", "")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/noninf_out"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

from examples.countdown.non_inferiority import HELDOUT  # noqa: E402


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from grpo_guard.adapters.gsm8k_reward import gsm8k_rule_verifier

    if not CKPT:
        raise SystemExit("GRPO_GUARD_CKPT required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    model.eval()
    correct = 0
    per_q = []
    with torch.no_grad():
        for q, ans in HELDOUT:
            ids = tokenizer(f"{q}\nAnswer with only the final number.", return_tensors="pt")["input_ids"]
            gen = model.generate(ids.to(model.device), do_sample=False, max_new_tokens=16)
            text = tokenizer.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
            comps = gsm8k_rule_verifier(text, float(ans))
            ok = comps["correctness"] == 1.0
            correct += 1 if ok else 0
            per_q.append({"q": q[:36], "golden": ans, "correct": ok})
    result = {"ckpt": CKPT, "n": len(HELDOUT), "correct": correct,
              "accuracy": correct / len(HELDOUT), "per_q": per_q}
    (OUT_DIR / "eval_heldout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
