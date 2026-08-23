#!/usr/bin/env bash
# Resume the interrupted P0-fixed RL training from its event log + checkpoints.
# D18 GPU1-only shared-card mode: trainer (SGD + checkpointing) and vLLM
# (mem_util 0.3) BOTH on GPU1 so we co-exist with agent-ttrl's GPU0 training.
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_LOOP_OUT=/root/autodl-tmp/grpo-guard/rl_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
CUDA_VISIBLE_DEVICES=1 \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/rl_training_loop.py --resume \
  > /root/autodl-tmp/grpo-guard/rl_resume.log 2>&1
' > /dev/null 2>&1 &
echo RL_RESUME_LAUNCHED
