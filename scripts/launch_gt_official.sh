#!/usr/bin/env bash
# P0-4 official-path run: 20 steps through the wrapped official TRL
# GRPOTrainer with pre-registered fault injection (guarded_trainer_official_run.py).
# Trainer sees GPU0 only; vLLM server on GPU1.
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_OUT=/root/autodl-tmp/grpo-guard/gt_official_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
GRPO_GUARD_STEPS=${GRPO_GUARD_STEPS:-20} \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/guarded_trainer_official_run.py \
  > /root/autodl-tmp/grpo-guard/gt_official.log 2>&1
' > /dev/null 2>&1 &
echo GT_OFFICIAL_LAUNCHED
