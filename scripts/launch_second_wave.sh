#!/usr/bin/env bash
# Launch the second-wave experiments on autodl2 (user-directed, D10).
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_OUT=/root/autodl-tmp/grpo-guard/second_wave_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/second_wave.py \
  > /root/autodl-tmp/grpo-guard/second_wave.log 2>&1
' > /dev/null 2>&1 &
echo SECOND_WAVE_LAUNCHED
