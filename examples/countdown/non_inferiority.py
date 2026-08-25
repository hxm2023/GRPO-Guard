"""E2: guard on/off non-inferiority (official TRL path, 5 seeds x 2 arms).

No fault injection: 30 official GRPO steps per run, guard on vs off.
Primary metric: frozen 16-prompt held-out accuracy (greedy decode of the
trained model vs the base model) — the guard must NOT degrade quality
beyond the pre-registered margin (|delta| <= 1/16 held-out question,
i.e. no worse than one question).

Secondary: in-train reward mean, step time (overhead).

Outputs: <out>/non_inferiority.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_ID = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/noninf_out"))
VLLM_HOST = os.environ.get("GRPO_GUARD_VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8012"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
N_STEPS = int(os.environ.get("GRPO_GUARD_STEPS", "30"))
VLLM_MEM = float(os.environ.get("GRPO_GUARD_VLLM_MEM", "0.25"))
ARM = os.environ.get("GRPO_GUARD_ARM", "on")
SEED = int(os.environ.get("GRPO_GUARD_SEED", "20260825"))
MAX_COMPLETION = int(os.environ.get("GRPO_GUARD_MAX_COMPLETION", "32"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

# 16 frozen held-out prompts — DISJOINT from the 8 training prompts
HELDOUT = [
    ("Emma has 15 stickers and gives 6 to her friend. How many stickers does she have left?", 9),
    ("A bakery makes 8 loaves every day. How many loaves in 5 days?", 40),
    ("A tree has 30 apples. 12 fall down. How many apples remain on the tree?", 18),
    ("Four friends share 20 candies equally. How many candies does each get?", 5),
    ("A library has 45 books and adds 15 more. How many books now?", 60),
    ("Lily buys 3 pencils at 2 yuan each. How much does she pay in total?", 6),
    ("A bike travels 12 km per hour. How far in 3 hours?", 36),
    ("There are 7 rows of chairs with 4 chairs each. How many chairs in total?", 28),
    ("A box has 24 oranges. 9 are eaten. How many oranges are left?", 15),
    ("Jack runs 3 km each morning for 6 days. How many km in total?", 18),
    ("A class has 32 students. 14 are girls. How many boys are there?", 18),
    ("A spider has 8 legs. How many legs do 5 spiders have?", 40),
    ("A movie is 90 minutes long. Half is over. How many minutes remain?", 45),
    ("Sue saves 5 yuan a day for 9 days. How much does she save?", 45),
    ("A garden has 18 tulips and 14 roses. How many flowers in total?", 32),
    ("A bottle holds 2 liters. How many liters in 7 bottles?", 14),
]


TRAIN_PROMPTS = [
    ("A farmer has 12 cows and buys 7 more. How many cows does he have?", 19),
    ("Sara has 4 bags with 3 apples in each. How many apples in total?", 12),
    ("A bus starts with 20 passengers. 5 get off and 8 get on. How many are on the bus?", 23),
    ("A rectangle is 6 units wide and 4 units tall. What is its area?", 24),
    ("Tom reads 5 pages a day for a whole week. How many pages does he read?", 35),
    ("There are 3 boxes with 6 pens each, and 4 extra pens. How many pens total?", 22),
    ("A shop sells 12 cupcakes on Monday and twice that on Tuesday. How many on Tuesday?", 24),
    ("A train covers 45 miles in 1 hour. How far does it go in 4 hours?", 180),
]


def build_train_dataset():
    from datasets import Dataset

    rows = [{"prompt": q + "\nAnswer with only the final number.", "answer": a}
            for q, a in TRAIN_PROMPTS]
    return Dataset.from_list(rows)


def reward_func_train(prompts, completions, **kwargs):
    from grpo_guard.adapters.gsm8k_reward import gsm8k_rule_verifier

    answers = kwargs.get("answer")
    scores = []
    for i, comp in enumerate(completions):
        ans = answers[i] if answers is not None else TRAIN_PROMPTS[i % 8][1]
        scores.append(gsm8k_rule_verifier(comp, float(ans))["correctness"])
    return scores


def heldout_eval(model, tokenizer) -> dict:
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
            ok = comps["correctness"] == 1.0
            correct += 1 if ok else 0
            per_q.append({"q": q[:36], "golden": ans, "correct": ok})
    model.train()
    return {"n": len(HELDOUT), "correct": correct, "accuracy": correct / len(HELDOUT),
            "per_q": per_q}


def main() -> int:
    import torch
    from trl import GRPOConfig, GRPOTrainer

    from grpo_guard.adapters.guarded_grpo_trainer import GuardedGRPOTrainer
    from grpo_guard.adapters.runtime_attest import (
        CANARY,
        drift,
        model_logprob_fingerprint,
        server_logprob_fingerprint,
    )

    from examples.countdown.closed_loop import start_server, stop_server
    from examples.countdown.smoke_train import _patch_device_normalization

    _patch_device_normalization()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    server = start_server(OUT_DIR / "vllm_server.log", port=VLLM_PORT, mem_util=VLLM_MEM,
                          device=os.environ.get("GRPO_GUARD_SERVER_DEVICE", "1"))

    args = GRPOConfig(
        output_dir=str(OUT_DIR / "ckpt"),
        run_name=f"noninf-{ARM}-s{SEED}",
        learning_rate=1e-6,
        beta=0.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        generation_batch_size=4,
        max_completion_length=MAX_COMPLETION,
        num_train_epochs=1,
        max_steps=N_STEPS,
        gradient_checkpointing=True,
        logging_steps=1,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=VLLM_HOST,
        vllm_server_port=VLLM_PORT,
        seed=SEED,
        log_completions=False,
    )

    class NonInfTrainer(GuardedGRPOTrainer, GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, guard_enabled=(ARM == "on"), **kwargs)
            self._per_step: list[dict] = []

        def training_step(self, model, inputs, num_items_in_batch):
            step = self.state.global_step
            t0 = time.perf_counter()
            out = super().training_step(model, inputs, num_items_in_batch)
            loss = float(out.detach().cpu()) if hasattr(out, "detach") else float(out)
            rewards = []
            for name in getattr(self, "reward_func_names", []):
                rewards.extend(self._logs.get("rewards", {}).get(name, []))
            self._per_step.append({
                "step": step, "loss": round(loss, 4),
                "reward_mean": round(float(sum(rewards) / max(1, len(rewards))), 4),
                "wall_s": round(time.perf_counter() - t0, 3),
            })
            return out

    trainer = NonInfTrainer(
        model=MODEL_ID,
        args=args,
        reward_funcs=reward_func_train,
        train_dataset=build_train_dataset(),
        guard_events_dir=OUT_DIR / "events",
        guard_store_dir=OUT_DIR / "store",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_eval = heldout_eval(trainer.model, tokenizer)  # base weights

    t0 = time.perf_counter()
    trainer.train()
    wall_s = time.perf_counter() - t0

    trained_eval = heldout_eval(trainer.model, tokenizer)

    result = {
        "experiment": "E2 guard on/off non-inferiority (official TRL path)",
        "arm": ARM, "seed": SEED, "steps": N_STEPS,
        "heldout_base_accuracy": base_eval["accuracy"],
        "heldout_trained_accuracy": trained_eval["accuracy"],
        "heldout_correct": trained_eval["correct"],
        "heldout_n": trained_eval["n"],
        "train_reward_mean": round(float(np.mean([p["reward_mean"] for p in trainer._per_step])), 4),
        "mean_step_time_s": round(float(np.mean([p["wall_s"] for p in trainer._per_step])), 4),
        "wall_time_s": round(wall_s, 1),
        "per_question": trained_eval["per_q"],
    }
    (OUT_DIR / "non_inferiority.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    stop_server(server)
    return 0


if __name__ == "__main__":
    sys.exit(main())
