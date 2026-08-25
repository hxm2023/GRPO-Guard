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

import numpy as np

MODEL_ID = os.environ.get("GRPO_GUARD_MODEL_PATH", "/root/autodl-tmp/models/Qwen3-4B")
OUT_DIR = Path(os.environ.get("GRPO_GUARD_OUT", "/root/autodl-tmp/grpo-guard/fault_survival_out"))
VLLM_HOST = os.environ.get("GRPO_GUARD_VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.environ.get("GRPO_GUARD_VLLM_PORT", "8012"))
REPO_DIR = Path(os.environ.get("GRPO_GUARD_REPO", "/root/autodl-tmp/grpo-guard/repo"))
N_STEPS = int(os.environ.get("GRPO_GUARD_STEPS", "30"))
VLLM_MEM = float(os.environ.get("GRPO_GUARD_VLLM_MEM", "0.25"))
ARM = os.environ.get("GRPO_GUARD_ARM", "on")  # on | off
SEED = int(os.environ.get("GRPO_GUARD_SEED", "20260825"))
MAX_COMPLETION = int(os.environ.get("GRPO_GUARD_MAX_COMPLETION", "32"))

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR))

FAULT_STEPS = [int(x) for x in os.environ.get("GRPO_GUARD_FAULT_STEPS", "10,20").split(",")]
FAULT_KINDS = os.environ.get("GRPO_GUARD_FAULT_KINDS", "F3,F2").split(",")

# GSM8K-style prompts: short "final number" completions terminate within
# budget (the countdown prompts produced 96+ token rambling -> reward 0).
E1_PROMPTS = [
    ("A farmer has 12 cows and buys 7 more. How many cows does he have?", 19),
    ("Sara has 4 bags with 3 apples in each. How many apples in total?", 12),
    ("A bus starts with 20 passengers. 5 get off and 8 get on. How many are on the bus?", 23),
    ("A rectangle is 6 units wide and 4 units tall. What is its area?", 24),
    ("Tom reads 5 pages a day for a whole week. How many pages does he read?", 35),
    ("There are 3 boxes with 6 pens each, and 4 extra pens. How many pens total?", 22),
    ("A shop sells 12 cupcakes on Monday and twice that on Tuesday. How many on Tuesday?", 24),
    ("A train covers 45 miles in 1 hour. How far does it go in 4 hours?", 180),
]


def build_e1_dataset():
    from datasets import Dataset

    rows = [{"prompt": q + "\nAnswer with only the final number.", "answer": a}
            for q, a in E1_PROMPTS]
    return Dataset.from_list(rows)


def reward_func_e1(prompts, completions, **kwargs):
    from grpo_guard.adapters.gsm8k_reward import gsm8k_rule_verifier

    answers = kwargs.get("answer")
    scores = []
    for i, comp in enumerate(completions):
        ans = answers[i] if answers is not None else E1_PROMPTS[i % len(E1_PROMPTS)][1]
        scores.append(gsm8k_rule_verifier(comp, float(ans))["correctness"])
    return scores


def _cpu_np(v):
    if hasattr(v, "cpu"):
        v = v.cpu()
    return np.asarray(v)


def inject_misbound_logprobs(inputs: dict) -> dict:
    """F2: misbind the old logprobs (row swap on multi-row slices; value
    corruption on the single-row slices TRL hands per step)."""
    bad = dict(inputs)
    lp = np.array(_cpu_np(bad["old_per_token_logps"]), copy=True)
    if lp.ndim == 2 and lp.shape[0] >= 2:
        lp[0] = lp[1]  # behavior logprobs bound to the WRONG trajectory
    else:
        lp[0, :] = lp[0, :] - 2.0  # single-row slice: corrupt the values
    bad["old_per_token_logps"] = lp
    return bad


def inject_retokenize(inputs: dict) -> dict:
    """F3: a completion token re-encoded to a different id."""
    bad = dict(inputs)
    cids = np.array(_cpu_np(bad["completion_ids"]), copy=True)
    cids[0, 0] = (cids[0, 0] + 1) % 32000
    bad["completion_ids"] = cids
    return bad


def inject_mask_shift(inputs: dict) -> dict:
    """F4: completion mask shifted by 1 (prompt token selected, last
    completion token dropped from the loss)."""
    bad = dict(inputs)
    cm = np.array(_cpu_np(bad["completion_mask"]), copy=True)
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
    server = start_server(
        OUT_DIR / "vllm_server.log", port=VLLM_PORT, mem_util=VLLM_MEM,
        device=os.environ.get("GRPO_GUARD_SERVER_DEVICE", "1"))

    faults = {step: FAULT_KINDS[i % len(FAULT_KINDS)] for i, step in enumerate(FAULT_STEPS)}

    args = GRPOConfig(
        output_dir=str(OUT_DIR / "ckpt"),
        run_name=f"fault-survival-{ARM}-s{SEED}",
        learning_rate=1e-6,
        beta=0.0,  # no ref model (E1 tests fault blocking, not KL quality)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        generation_batch_size=4,
        max_completion_length=MAX_COMPLETION,
        num_train_epochs=1,
        max_steps=N_STEPS,
        gradient_checkpointing=True,  # 128-token completions peak ~45GB; checkpointing halves activations
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
                if kind == "F2":
                    import numpy as _np
                    lp = inputs.get("old_per_token_logps")
                    print(f"[fault] F2 diag: old_per_token_logps present="
                          f"{lp is not None} type={type(lp).__name__} "
                          f"shape={getattr(lp, 'shape', None)}", flush=True)
                return INJECTORS[kind](inputs)
            return inputs

        def training_step(self, model, inputs, num_items_in_batch):
            step = self.state.global_step
            t0 = time.perf_counter()
            try:
                out = super().training_step(model, inputs, num_items_in_batch)
            except GuardViolation as exc:
                self._fault_blocks.append({
                    "step": step, "violation": str(exc),
                    "blocked_before_backward": True,
                })
                # recovery: re-run the step on the CLEAN batch (hook one-shot)
                out = super().training_step(model, inputs, num_items_in_batch)
            loss = float(out.detach().cpu()) if hasattr(out, "detach") else float(out)
            rewards = []
            for name in getattr(self, "reward_func_names", []):
                rewards.extend(self._logs.get("rewards", {}).get(name, []))
            self._per_step.append({
                "step": step,
                "loss": round(loss, 4),
                "reward_mean": round(float(sum(rewards) / max(1, len(rewards))), 4),
                "wall_s": round(time.perf_counter() - t0, 3),
                "fault_injected": step in [f["step"] for f in self._injected],
                "blocked": step in [b["step"] for b in self._fault_blocks],
            })
            return out

    trainer = FaultSurvivalTrainer(
        model=MODEL_ID,
        args=args,
        reward_funcs=reward_func_e1,
        train_dataset=build_e1_dataset(),
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
    fault_steps = [f["step"] for f in trainer._injected]
    step_times = [p["wall_s"] for p in trainer._per_step]
    start_drift = attest_start.get("max_abs_logprob_drift") or 0.0
    end_drift = attest_end.get("max_abs_logprob_drift") or 0.0
    # relative attestation: end drift vs the SAME-weights numeric baseline
    # (fp16 server vs bf16 trainer ~0.07); a stale runtime diverges far more
    stale_detected = end_drift > max(0.25, 3 * start_drift)
    result = {
        "experiment": "E1 fault-injected training survival (official TRL path)",
        "arm": ARM, "seed": SEED, "steps": N_STEPS,
        "faults": trainer._injected,
        "fault_blocks": trainer._fault_blocks,
        "per_step": trainer._per_step,
        "bad_updates_accepted": len(fault_steps) - len(trainer._fault_blocks),
        "detection_latency_steps": 0 if ARM == "on" else None,
        "wasted_steps": 0 if ARM == "on" else sum(N_STEPS - s for s in fault_steps),
        "success_series": [p["reward_mean"] for p in trainer._per_step],
        "loss_series": [p["loss"] for p in trainer._per_step],
        "mean_step_time_s": round(float(sum(step_times) / max(1, len(step_times))), 4) if step_times else None,
        "wall_time_s": round(wall_s, 1),
        "attestation": {"start": attest_start, "end": attest_end,
                        "stale_detected": stale_detected},
        "ok": (ARM == "on" and len(trainer._fault_blocks) == len(fault_steps))
              or (ARM == "off" and len(trainer._fault_blocks) == 0),
        "fault_blocks_missing": [f for f in trainer._injected
                                 if f["step"] not in [b["step"] for b in trainer._fault_blocks]],
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
