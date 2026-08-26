"""F1 end-to-end: static-rollout detection on the OFFICIAL TRL path.

The ORIGINAL incident: the trainer updates weights but the rollout server
keeps serving an OLD policy. On the official path TRL syncs weights to
vLLM via ``vllm_generation.sync_weights`` after each step; this runner
freezes the sync at step K (pushing the BASE weights instead of the
trained ones) — the server then generates with the stale policy.

Detection: the dual-source runtime attestation (server
/get_sequence_logprobs fingerprint vs the trainer's forward on the same
frozen canary sequences) is taken at start, AFTER the freeze, and at the
end. A stale server diverges from the trainer → STALE_RUNTIME_SUSPECTED.

Metrics: fingerprints + verdicts, sync_calls after freeze, per-step
loss/reward (the stale-server rollouts should distort the update), and
wall time.

Outputs: <out>/f1_static_rollout.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_ID = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/f1_out"))
VLLM_HOST = os.environ.get("GRPO_GUARD_VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8012"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
N_STEPS = int(os.environ.get("GRPO_GUARD_STEPS", "30"))
VLLM_MEM = float(os.environ.get("GRPO_GUARD_VLLM_MEM", "0.2"))
SEED = int(os.environ.get("GRPO_GUARD_SEED", "20260825"))
MAX_COMPLETION = int(os.environ.get("GRPO_GUARD_MAX_COMPLETION", "32"))
FREEZE_AFTER = int(os.environ.get("GRPO_GUARD_FREEZE_AFTER", "15"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

from examples.countdown.non_inferiority import TRAIN_PROMPTS, build_train_dataset, reward_func_train  # noqa: E402


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
        run_name=f"f1-static-rollout-s{SEED}",
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

    class F1Trainer(GuardedGRPOTrainer, GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._per_step: list[dict] = []
            self._frozen_sync_calls = 0

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

    trainer = F1Trainer(
        model=MODEL_ID,
        args=args,
        reward_funcs=reward_func_train,
        train_dataset=build_train_dataset(),
        guard_events_dir=OUT_DIR / "events",
        guard_store_dir=OUT_DIR / "store",
    )

    def attest(tag: str) -> dict:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        sequences, prompt_lengths = [], []
        for prompt, _ in CANARY:
            pids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            comp = tokenizer(" the answer is", add_special_tokens=False)["input_ids"][:8]
            sequences.append(pids + comp)
            prompt_lengths.append(len(pids))
        try:
            server_fp = server_logprob_fingerprint(VLLM_HOST, VLLM_PORT, sequences, prompt_lengths)
        except Exception as exc:
            return {"tag": tag, "error": str(exc)[:160]}
        model_fp = model_logprob_fingerprint(trainer.model, tokenizer, sequences, prompt_lengths)
        return {"tag": tag, "server": server_fp["digest"][:16], "model": model_fp["digest"][:16],
                **drift(server_fp, model_fp)}

    # F1 injection: freeze the sync after step FREEZE_AFTER (push BASE weights)
    base_state = {name: p.detach().cpu().clone()
                  for name, p in trainer.model.named_parameters()}
    orig_sync = trainer.vllm_generation.sync_weights

    def frozen_sync():
        if trainer.state.global_step > FREEZE_AFTER:
            trainer._frozen_sync_calls += 1
            for name, param in trainer.model.named_parameters():
                name_fixed = trainer.vllm_generation._fix_param_name_to_vllm(name)
                trainer.vllm_generation._push_param_to_vllm(
                    name_fixed, base_state[name].to(param.device))
        else:
            orig_sync()

    trainer.vllm_generation.sync_weights = frozen_sync

    attestations = [attest("start")]
    t0 = time.perf_counter()
    trainer.train()
    wall_s = time.perf_counter() - t0
    attestations.append(attest("after_freeze"))
    attestations.append(attest("end"))

    # relative verdict: end drift vs the same-weights numeric baseline
    # (fp16 server vs bf16 trainer ~0.06); a frozen server diverges far more
    base_drift = attestations[0].get("max_abs_logprob_drift") or 0.0
    max_late_drift = max((a.get("max_abs_logprob_drift") or 0.0) for a in attestations[1:])
    # server digest equality is NOT reliable across calls (vLLM kernel
    # noise); the drift signal scales with how much training moved the
    # weights. Verdict: the freeze happened AND the server-trainer
    # divergence grew beyond 2x the same-weights numeric baseline.
    stale_detected = max_late_drift > 2 * base_drift  # drift-only (no self-witness)
    result = {
        "experiment": "F1 static-rollout end-to-end (official TRL path)",
        "seed": SEED, "steps": N_STEPS, "freeze_after": FREEZE_AFTER,
        "frozen_sync_calls": trainer._frozen_sync_calls,
        "stale_detected": stale_detected,
        "stale_detection_drift": round(max_late_drift, 6),
        "stale_detection_baseline": round(base_drift, 6),
        "attestations": attestations,
        "pre_freeze_reward_mean": round(float(np.mean([p["reward_mean"] for p in trainer._per_step
                                                       if p["step"] <= FREEZE_AFTER])), 4),
        "post_freeze_reward_mean": round(float(np.mean([p["reward_mean"] for p in trainer._per_step
                                                        if p["step"] > FREEZE_AFTER])), 4),
        "per_step": trainer._per_step,
        "wall_time_s": round(wall_s, 1),
    }
    (OUT_DIR / "f1_static_rollout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    stop_server(server)
    return 0


if __name__ == "__main__":
    sys.exit(main())
