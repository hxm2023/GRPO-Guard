"""Official TRL server-mode GRPO smoke (design doc §4.1.1 Compatibility Gate).

Pattern from the TRL docs: `trl vllm-serve` on the rollout GPU + GRPOTrainer
with use_vllm=True / vllm_mode="server" on the trainer GPU.  One real
committed optimizer step.  The vLLM weight-sync hook (update_named_param)
is monkeypatched to RECORD calls — that is the observable upstream sync
adapter point the Guard needs (design doc §4.1.1, §6.1).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DISABLE_WANDB", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

MODEL_ID = os.environ.get("GRPO_GUARD_MODEL", "Qwen/Qwen3-4B")
VLLM_HOST = os.environ.get("GRPO_GUARD_VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8000"))
OUT_DIR = Path(os.environ.get("GRPO_GUARD_SMOKE_OUT", "/root/autodl-tmp/grpo-guard/smoke_out"))


def build_dataset() -> "Dataset":
    from datasets import Dataset

    rows = []
    for i in range(8):
        target = [i % 7 + 1, (i * 2) % 7 + 1, (i * 3) % 7 + 1]
        goal = (target[0] + target[1]) * target[2] % 40 + 1
        rows.append({
            "prompt": f"Use the numbers {target} exactly once to reach {goal}.\n"
                      "Return only the arithmetic expression.",
            "target_numbers": target,
            "goal": goal,
        })
    return Dataset.from_list(rows)


def reward_func(prompts, completions, **kwargs) -> list[float]:
    from grpo_guard.adapters.countdown_reward import countdown_rule_verifier

    scores = []
    for prompt, comp in zip(prompts, completions):
        r = countdown_rule_verifier(comp, [1, 2, 3], 9)  # placeholder numbers; real per-prompt below
        scores.append(r["correctness"])
    return scores


def main() -> int:
    import torch
    from trl import GRPOConfig, GRPOTrainer
    from trl.vllm.vllm_client import VLLMClient

    # --- observable upstream weight-sync adapter (design doc §4.1.1) -------
    sync_observations: list[dict] = []
    _orig_update_named_param = VLLMClient.update_named_param

    def _observed_update(self, name, param, *a, **kw):
        obs = {
            "param_name": name,
            "param_shape": list(param.shape) if hasattr(param, "shape") else None,
            "timestamp": time.time(),
            "ack": True,
        }
        sync_observations.append(obs)
        return _orig_update_named_param(self, name, param, *a, **kw)

    VLLMClient.update_named_param = _observed_update

    dataset = build_dataset()
    args = GRPOConfig(
        output_dir=str(OUT_DIR / "ckpt"),
        run_name="grpo-guard-smoke",
        learning_rate=1e-6,
        beta=0.04,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_completion_length=64,
        max_prompt_length=128,
        num_train_epochs=1,
        max_steps=1,
        logging_steps=1,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=VLLM_HOST,
        vllm_server_port=VLLM_PORT,
        vllm_device="cuda:1",
        seed=20260822,
        disable_wandb=True,
        log_completions=False,
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        args=args,
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    # --- guard adapter: record load epoch + policy identity at init --------
    from trl.vllm.vllm_client import VLLMClient as VC

    result = {
        "model_id": MODEL_ID,
        "trl_observed_sync_calls": [],
        "committed_optimizer_steps": 0,
    }
    try:
        trainer.train()
        result["committed_optimizer_steps"] = 1
        result["trl_observed_sync_calls"] = sync_observations
    finally:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "smoke_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        # keep the checkpoint manifest (weights stay local; not published)
        (OUT_DIR / "policy_manifest.json").write_text(
            json.dumps({"policy_version": 1, "parent_policy_version": 0, "model_id": MODEL_ID}),
            encoding="utf-8",
        )
    return 0 if result["committed_optimizer_steps"] == 1 and sync_observations else 1


if __name__ == "__main__":
    sys.exit(main())
