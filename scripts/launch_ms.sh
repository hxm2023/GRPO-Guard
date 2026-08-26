#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_LOOP_OUT=/root/autodl-tmp/grpo-guard/multi_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/multi_step_loop.py \
  > /root/autodl-tmp/grpo-guard/multi.log 2>&1
' > /dev/null 2>&1 &
echo MULTI_LAUNCHED
