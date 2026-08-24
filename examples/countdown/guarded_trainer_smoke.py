"""P1-1 smoke: the OFFICIAL TRL GRPOTrainer wrapped by GuardedGRPOTrainer.

One real server-mode training step through the wrapped path:
rollout seam (align + logprob contract checks, generation events),
step seam (pre-update consistency), commit seam (content-hashed
checkpoint).  Outputs: <out>/guarded_trainer_smoke.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODEL_ID = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/gt_smoke_out"))
VLLM_HOST = os.environ.get("GRPO_GUARD_VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8012"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

from examples.countdown.smoke_train import build_dataset, reward_func  # noqa: E402


def main() -> int:
    import torch
    from trl import GRPOConfig, GRPOTrainer
    from trl.generation.vllm_client import VLLMClient

    from grpo_guard.adapters.guarded_grpo_trainer import GuardedGRPOTrainer

    from examples.countdown.closed_loop import start_server, stop_server
    from examples.countdown.smoke_train import _patch_device_normalization

    _patch_device_normalization()
    server = start_server(OUT_DIR / "vllm_server.log", port=VLLM_PORT, mem_util=0.4, device="1")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = GRPOConfig(
        output_dir=str(OUT_DIR / "ckpt"),
        run_name="grpo-guard-guarded-smoke",
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

    class GuardedTRLTrainer(GuardedGRPOTrainer, GRPOTrainer):
        pass

    trainer = GuardedTRLTrainer(
        model=MODEL_ID,
        args=args,
        reward_funcs=reward_func,
        train_dataset=build_dataset(),
        guard_events_dir=OUT_DIR / "events",
        guard_store_dir=OUT_DIR / "store",
    )

    trainer.train()

    # verify the guard seams fired
    events = list(trainer._guard_log.iterate())
    gen_events = [e for e in events if e["event_type"] == "generation_finished"]
    result = {
        "scope": "P1-1: official TRL GRPOTrainer wrapped by GuardedGRPOTrainer "
                 "(rollout/step/commit seams), 1 real server-mode step",
        "guard_generation_events": len(gen_events),
        "guard_rollouts_recorded": len(trainer._guard_rollouts),
        "guard_commit_sha256": getattr(trainer, "_last_guard_commit_sha256", None),
        "guard_violations_raised": 0,
    }
    (OUT_DIR / "guarded_trainer_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    stop_server(server)
    return 0 if gen_events else 1


if __name__ == "__main__":
    sys.exit(main())
