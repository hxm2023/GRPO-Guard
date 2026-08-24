#!/usr/bin/env bash
# P1-2 guard-off comparison arm: same RL loop WITHOUT the guard
# (synthetic ALLOW decisions, plain loss/step) — used to show the guard
# does not degrade normal training.  Compare against the guard-on 20-step
# run (artifacts/v0.1.0/rl_training_final/).
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_LOOP_OUT=/root/autodl-tmp/grpo-guard/guard_off_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/rl_training_loop.py --guard-off --steps 10 \
  > /root/autodl-tmp/grpo-guard/guard_off.log 2>&1
' > /dev/null 2>&1 &
echo GUARD_OFF_LAUNCHED
