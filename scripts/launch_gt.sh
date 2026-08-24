#!/usr/bin/env bash
# P1-1 GuardedGRPOTrainer smoke: trainer sees GPU0 only (accelerator
# device cuda:0), vLLM server on GPU1 (start_server device=1).
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_OUT=/root/autodl-tmp/grpo-guard/gt_smoke_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/guarded_trainer_smoke.py \
  > /root/autodl-tmp/grpo-guard/gt_smoke.log 2>&1
' > /dev/null 2>&1 &
echo GT_LAUNCHED
