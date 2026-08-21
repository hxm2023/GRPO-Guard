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

os.environ.setdefault("WANDB_DISABLED", "1")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

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

    targets = kwargs.get("target_numbers")
    goals = kwargs.get("goal")
    scores = []
    for i, comp in enumerate(completions):
        tgt = targets[i] if targets is not None else [1, 2, 3]
        goal = goals[i] if goals is not None else 9
        scores.append(countdown_rule_verifier(comp, tgt, goal)["correctness"])
    return scores


def _patch_device_normalization() -> None:
    """Adapter fix for trl 1.10.0 + vllm 0.26.0 (design doc §14.2).

    VLLMClient.init_communicator passes ``accelerator.device`` (unindexed
    torch.device('cuda')) into vLLM's PyNcclCommunicator, whose init warm-up
    all_reduce asserts in_tensor.device == self.device and crashes on
    'cuda' != 'cuda:0'.  Normalizing to the current device index is small,
    local, and version-guarded; the patch fails closed (asserts versions)
    instead of silently skipping.
    """
    import torch
    import trl
    import vllm
    from trl.generation.vllm_client import VLLMClient

    assert trl.__version__ == "1.10.0", f"patch built for trl 1.10.0, got {trl.__version__}"
    assert vllm.__version__ == "0.26.0", f"patch built for vllm 0.26.0, got {vllm.__version__}"

    _orig = VLLMClient.init_communicator

    def _normalized(self, device, *a, **kw):
        if isinstance(device, torch.device) and device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        return _orig(self, device, *a, **kw)

    VLLMClient.init_communicator = _normalized


def main() -> int:
    import torch
    from trl import GRPOConfig, GRPOTrainer
    from trl.generation.vllm_client import VLLMClient

    _patch_device_normalization()

    # --- observable upstream weight-sync adapter (design doc §4.1.1) -------
    sync_observations: list[dict] = []
    _orig_update_named_param = VLLMClient.update_named_param

    def _observed_update(self, name, weights, *a, **kw):
        obs = {
            "param_name": name,
            "param_shape": list(weights.shape) if hasattr(weights, "shape") else None,
            "timestamp": time.time(),
            "ack": True,
        }
        sync_observations.append(obs)
        return _orig_update_named_param(self, name, weights, *a, **kw)

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
        generation_batch_size=4,
        max_completion_length=64,
        num_train_epochs=1,
        max_steps=1,
        logging_steps=1,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=VLLM_HOST,
        vllm_server_port=VLLM_PORT,
        seed=20260822,
        log_completions=False,
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        args=args,
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

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
        (OUT_DIR / "policy_manifest.json").write_text(
            json.dumps({"policy_version": 1, "parent_policy_version": 0, "model_id": MODEL_ID}),
            encoding="utf-8",
        )
    return 0 if result["committed_optimizer_steps"] == 1 and sync_observations else 1


if __name__ == "__main__":
    sys.exit(main())
