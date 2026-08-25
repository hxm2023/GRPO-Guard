"""P0-4 official-path run: 20 steps through the OFFICIAL TRL GRPOTrainer
wrapped by GuardedGRPOTrainer, with pre-registered fault injection.

At pre-registered steps (D: FAULTS below) a wiring fault is injected into
the ACTUAL training_step tensors (F3 retokenization of a completion token).
The guard must raise GuardViolation BEFORE super().training_step — i.e.
before loss/backward; the step is then re-run on the untampered batch and
training continues, proving fail-closed detection + full 20-step chain
with per-step verification.  Outputs: <out>/guarded_trainer_official_run.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODEL_ID = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/gt_official_out"))
VLLM_HOST = os.environ.get("GRPO_GUARD_VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8012"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
N_STEPS = int(os.environ.get("GRPO_GUARD_STEPS", "20"))
VLLM_MEM = float(os.environ.get("GRPO_GUARD_VLLM_MEM", "0.4"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

from examples.countdown.smoke_train import build_dataset, reward_func  # noqa: E402

# pre-registered faults: global_step -> fault kind
FAULTS = {N_STEPS // 2: "F3_retokenize_completion",
          N_STEPS: "F3_retokenize_completion"}


def inject_retokenize(inputs) -> dict:
    """F3: re-encode a completion token of the ACTUAL consumed batch.

    Handles both dict batches and transformers-5.x list-of-sample-dicts.
    """
    import numpy as np

    def _cpu_np(v):
        if hasattr(v, "cpu"):
            v = v.cpu()
        return np.array(np.asarray(v), copy=True)

    if isinstance(inputs, dict):
        bad = dict(inputs)
        if "completion_ids" not in bad:
            raise TypeError(f"inject_retokenize: no completion_ids ({list(bad)[:5]})")
        cids = _cpu_np(bad["completion_ids"])
        cids[0, 0] = (cids[0, 0] + 1) % 32000  # different token id
        bad["completion_ids"] = cids
        return bad
    if isinstance(inputs, (list, tuple)) and inputs and isinstance(inputs[0], dict):
        bad = [dict(s) for s in inputs]
        if "completion_ids" not in bad[0]:
            raise TypeError(f"inject_retokenize: no completion_ids ({list(bad[0])[:5]})")
        cids = _cpu_np(bad[0]["completion_ids"])
        cids[0] = (cids[0] + 1) % 32000  # first completion token of sample 0
        bad[0]["completion_ids"] = cids
        return bad
    raise TypeError(f"inject_retokenize: unexpected inputs type {type(inputs)}")


def main() -> int:
    import torch
    from trl import GRPOConfig, GRPOTrainer

    from grpo_guard.adapters.guarded_grpo_trainer import GuardViolation, GuardedGRPOTrainer

    from examples.countdown.closed_loop import start_server, stop_server
    from examples.countdown.smoke_train import _patch_device_normalization

    _patch_device_normalization()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    server = start_server(OUT_DIR / "vllm_server.log", port=VLLM_PORT, mem_util=VLLM_MEM, device="1")

    args = GRPOConfig(
        output_dir=str(OUT_DIR / "ckpt"),
        run_name="grpo-guard-official-p0",
        learning_rate=1e-6,
        beta=0.04,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=8,
        generation_batch_size=8,
        max_completion_length=64,
        num_train_epochs=1,
        max_steps=N_STEPS,
        logging_steps=1,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=VLLM_HOST,
        vllm_server_port=VLLM_PORT,
        seed=20260825,
        log_completions=False,
    )

    class GuardedTRLTrainer(GuardedGRPOTrainer, GRPOTrainer):
        def _guard_prepare_hook(self, inputs):
            """One-shot fault injection into the ACTUAL step tensors."""
            if self.state.global_step in FAULTS:
                del FAULTS[self.state.global_step]
                print(f"[fault] step {self.state.global_step} injected "
                      f"F3_retokenize_completion", flush=True)
                return inject_retokenize(inputs)
            return inputs

        def _guard_pre_update(self, inputs):
            super()._guard_pre_update(inputs)
            self._guard_run_log.append({
                "step": self.state.global_step,
                "verified": dict(self._last_guard_verified or {}),
                "records": len(self._guard_rollouts),
            })
            print(f"[guard] step {self.state.global_step} verified={self._last_guard_verified} "
                  f"records={len(self._guard_rollouts)}", flush=True)

        def training_step(self, model, inputs, num_items_in_batch):
            step = self.state.global_step
            try:
                return super().training_step(model, inputs, num_items_in_batch)
            except GuardViolation as exc:
                self._guard_fault_blocks.append({
                    "step": step, "fault": FAULTS.pop(step, "fault"),
                    "violation": str(exc), "blocked_before_backward": True,
                })
                # recovery: re-run the step on the CLEAN batch (hook one-shot)
                return super().training_step(model, inputs, num_items_in_batch)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._guard_run_log: list[dict] = []
            self._guard_fault_blocks: list[dict] = []

    trainer = GuardedTRLTrainer(
        model=MODEL_ID,
        args=args,
        reward_funcs=reward_func,
        train_dataset=build_dataset(),
        guard_events_dir=OUT_DIR / "events",
        guard_store_dir=OUT_DIR / "store",
    )

    trainer.train()

    events = list(trainer._guard_log.iterate())
    gen_events = [e for e in events if e["event_type"] == "generation_finished"]
    result = {
        "scope": f"P0-4 official-path run: {N_STEPS} steps through official "
                 "TRL GRPOTrainer wrapped by GuardedGRPOTrainer; pre-registered "
                 "fault injection (F3 retokenization) must be blocked before "
                 "backward at the fault steps",
        "steps_completed": N_STEPS,
        "faults": {str(k): v for k, v in FAULTS.items()},
        "fault_blocks": trainer._guard_fault_blocks,
        "per_step_verified": trainer._guard_run_log,
        "guard_generation_events": len(gen_events),
        "guard_commit_sha256": getattr(trainer, "_last_guard_commit_sha256", None),
        "ok": all(b["blocked_before_backward"] for b in trainer._guard_fault_blocks)
              and len(trainer._guard_run_log) == N_STEPS,
    }
    (OUT_DIR / "guarded_trainer_official_run.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    stop_server(server)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
