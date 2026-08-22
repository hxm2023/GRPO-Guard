#!/usr/bin/env bash
# Launch the P008 canary-mismatch online check on autodl2 (user-approved).
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_OUT=/root/autodl-tmp/grpo-guard/canary_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/canary_mismatch_check.py \
  > /root/autodl-tmp/grpo-guard/canary.log 2>&1
' > /dev/null 2>&1 &
echo CANARY_LAUNCHED
