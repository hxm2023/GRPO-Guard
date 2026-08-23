#!/usr/bin/env bash
# Resume the interrupted P0-fixed RL training from its event log + checkpoints.
# The event log (rl_out/events) and checkpoints (rl_out/ckpt_v*) were preserved
# when agent-ttrl's GPU contention interrupted the run at step 2 (D18).
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_LOOP_OUT=/root/autodl-tmp/grpo-guard/rl_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/rl_training_loop.py --resume \
  > /root/autodl-tmp/grpo-guard/rl_resume.log 2>&1
' > /dev/null 2>&1 &
echo RL_RESUME_LAUNCHED
