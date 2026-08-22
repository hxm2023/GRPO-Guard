#!/usr/bin/env bash
# Launch the canary stress check on autodl2 (D11).
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_OUT=/root/autodl-tmp/grpo-guard/canary_stress_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/canary_stress.py \
  > /root/autodl-tmp/grpo-guard/canary_stress.log 2>&1
' > /dev/null 2>&1 &
echo CANARY_STRESS_LAUNCHED
