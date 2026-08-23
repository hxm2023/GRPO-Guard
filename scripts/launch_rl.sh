#!/usr/bin/env bash
# Launch the real RL training loop on autodl2 (D15).
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_LOOP_OUT=/root/autodl-tmp/grpo-guard/rl_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/rl_training_loop.py \
  > /root/autodl-tmp/grpo-guard/rl.log 2>&1
' > /dev/null 2>&1 &
echo RL_LAUNCHED
