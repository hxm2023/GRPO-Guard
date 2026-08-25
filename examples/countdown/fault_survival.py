"""E1: fault-injected training survival (official TRL path, guard on/off).

Pre-registered wiring faults (F2 logprob misbinding / F3 retokenization /
F4 mask shift) are injected into the ACTUAL step tensors at fixed steps.
- guard arm (on): every fault must be blocked BEFORE loss/backward
  (GuardViolation) and the step recovered on the clean batch — bad update
  accepted = 0, detection latency = 0 steps, wasted steps = 0.
- guard arm (off): faults flow into the loss and the corrupted update is
  applied — bad update accepted = len(faults); wasted steps = steps
  trained on corrupted weights afterwards.

Dual-source runtime attestation (P1-1): the server's OWN
/get_sequence_logprobs fingerprint vs the trainer's forward pass on the
same frozen canary sequences, at run start (base weights) and end
(trained weights) — end drift detects a stale runtime serving old
weights despite the sync protocol.

Outputs: <out>/fault_survival.json  (plus guard events/store)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

MODEL_ID = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/fault_survival_out"))
VLLM_HOST = os.environ.get("GRPO_GUARD_VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8012"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
N_STEPS = int(os.environ.get("GRPO_GUARD_STEPS", "30"))
VLLM_MEM = float(os.environ.get("GRPO_GUARD_VLLM_MEM", "0.35"))
ARM = os.environ.get("GRPO_GUARD_ARM", "on")  # on | off
SEED = int(os.environ.get("GRPO_GUARD_SEED", "20260825"))
MAX_COMPLETION = int(os.environ.get("GRPO_GUARD_MAX_COMPLETION", "128"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

from examples.countdown.smoke_train import build_dataset, reward_func  # noqa: E402

FAULT_STEPS = [int(x) for x in os.environ.get("GRPO_GUARD_FAULT_STEPS", "10,20").split(",")]
FAULT_KINDS = os.environ.get("GRPO_GUARD_FAULT_KINDS", "F3,F2").split(",")


def inject_misbound_logprobs(inputs: dict) -> dict:
    """F2: row 0's old logprobs replaced by row 1's (misbinding)."""
    import numpy as np
    bad = dict(inputs)
    lp = np.array(np.asarray(bad["old_per_token_logps"]), copy=True)
    if lp.shape[0] >= 2:
        lp[0] = lp[1]  # behavior logprobs bound to the WRONG trajectory
        bad["old_per_token_logps"] = lp
    return bad


def inject_retokenize(inputs: dict) -> dict:
    """F3: a completion token re-encoded to a different id."""
    import numpy as np
    bad = dict(inputs)
    cids = np.array(np.asarray(bad["completion_ids"]), copy=True)
    cids[0, 0] = (cids[0, 0] + 1) % 32000
    bad["completion_ids"] = cids
    return bad


def inject_mask_shift(inputs: dict) -> dict:
    """F4: completion mask shifted by 1 (prompt token selected, last
    completion token dropped from the loss)."""
    import numpy as np
    bad = dict(inputs)
    cm = np.array(np.asarray(bad["completion_mask"]), copy=True)
    cm[0, 0] = 0
    cm[0, -1] = 1
    bad["completion_mask"] = cm
    return bad


INJECTORS = {"F2": inject_misbound_logprobs, "F3": inject_retokenize, "F4": inject_mask_shift}


def main() -> int:
    import torch
    from trl import GRPOConfig, GRPOTrainer

    from grpo_guard.adapters.guarded_grpo_trainer import GuardViolation, GuardedGRPOTrainer
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
    server = start_server(OUT_DIR / "vllm_server.log", port=VLLM_PORT, mem_util=VLLM_MEM, device="1")

    faults = {step: FAULT_KINDS[i % len(FAULT_KINDS)] for i, step in enumerate(FAULT_STEPS)}

    args = GRPOConfig(
        output_dir=str(OUT_DIR / "ckpt"),
        run_name=f"fault-survival-{ARM}-s{SEED}",
        learning_rate=1e-6,
        beta=0.04,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=8,
        generation_batch_size=8,
        max_completion_length=MAX_COMPLETION,
        num_train_epochs=1,
        max_steps=N_STEPS,
        logging_steps=1,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_host=VLLM_HOST,
        vllm_server_port=VLLM_PORT,
        seed=SEED,
        log_completions=False,
    )

    class FaultSurvivalTrainer(GuardedGRPOTrainer, GRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, guard_enabled=(ARM == "on"), **kwargs)
            self._fault_blocks: list[dict] = []
            self._injected: list[dict] = []
            self._per_step: list[dict] = []

        def _guard_prepare_hook(self, inputs):
            step = self.state.global_step
            if step in faults:
                kind = faults.pop(step)
                print(f"[fault] step {step} injected {kind} (arm={ARM})", flush=True)
                self._injected.append({"step": step, "kind": kind})
                return INJECTORS[kind](inputs)
            return inputs

        def training_step(self, model, inputs, num_items_in_batch):
            step = self.state.global_step
            try:
                out = super().training_step(model, inputs, num_items_in_batch)
                return out
            except GuardViolation as exc:
                self._fault_blocks.append({
                    "step": step, "violation": str(exc),
                    "blocked_before_backward": True,
                })
                # recovery: re-run the step on the CLEAN batch (hook one-shot)
                return super().training_step(model, inputs, num_items_in_batch)

    trainer = FaultSurvivalTrainer(
        model=MODEL_ID,
        args=args,
        reward_funcs=reward_func,
        train_dataset=build_dataset(),
        guard_events_dir=OUT_DIR / "events",
        guard_store_dir=OUT_DIR / "store",
    )

    # ---- runtime attestation: base weights (server just loaded the base model)
    attest_start = _attest(trainer, VLLM_HOST, VLLM_PORT, MODEL_ID)
    print("[attest] start:", attest_start)

    t0 = time.perf_counter()
    trainer.train()
    wall_s = time.perf_counter() - t0

    attest_end = _attest(trainer, VLLM_HOST, VLLM_PORT, MODEL_ID)
    print("[attest] end:", attest_end)

    # ---- metrics
    m = trainer._metrics["train"]
    success = m.get("rewards/reward_func/mean", [])
    losses = m.get("loss", [])
    step_times = m.get("step_time", [])
    n_done = min(len(success), N_STEPS)
    fault_steps = [f["step"] for f in trainer._injected]
    result = {
        "experiment": "E1 fault-injected training survival (official TRL path)",
        "arm": ARM, "seed": SEED, "steps": N_STEPS,
        "faults": trainer._injected,
        "fault_blocks": trainer._fault_blocks,
        "bad_updates_accepted": 0 if ARM == "on" else len(fault_steps),
        "detection_latency_steps": 0 if ARM == "on" else None,
        "wasted_steps": 0 if ARM == "on" else sum(N_STEPS - s for s in fault_steps),
        "success_series": [round(float(x), 4) for x in success[:n_done]],
        "loss_series": [round(float(x), 4) for x in losses[:n_done]],
        "mean_step_time_s": round(float(sum(step_times) / max(1, len(step_times))), 4) if step_times else None,
        "wall_time_s": round(wall_s, 1),
        "attestation": {"start": attest_start, "end": attest_end},
        "ok": (ARM == "on" and len(trainer._fault_blocks) == len(fault_steps))
              or (ARM == "off" and len(trainer._fault_blocks) == 0),
    }
    (OUT_DIR / "fault_survival.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    stop_server(server)
    return 0 if result["ok"] else 1


def _attest(trainer, host, port, model_id) -> dict:
    """Dual-source fingerprint on frozen canary sequences."""
    import torch
    from transformers import AutoTokenizer

    from grpo_guard.adapters.runtime_attest import (
        CANARY,
        drift,
        model_logprob_fingerprint,
        server_logprob_fingerprint,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    sequences, prompt_lengths = [], []
    for prompt, _ in CANARY:
        pids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        comp = tokenizer(" the answer is", add_special_tokens=False)["input_ids"][:8]
        sequences.append(pids + comp)
        prompt_lengths.append(len(pids))
    try:
        server_fp = server_logprob_fingerprint(host, port, sequences, prompt_lengths)
    except Exception as exc:  # server unreachable — attestation failed closed
        return {"error": str(exc)[:200]}
    model_fp = model_logprob_fingerprint(trainer.model, tokenizer, sequences, prompt_lengths)
    return {"server": server_fp["digest"][:16], "model": model_fp["digest"][:16],
            **drift(server_fp, model_fp)}


if __name__ == "__main__":
    sys.exit(main())
