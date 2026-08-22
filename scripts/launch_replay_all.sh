#!/usr/bin/env bash
# Launch the full-group paired gradient replay on autodl2 (user-directed).
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_LOOP_DIR=/root/autodl-tmp/grpo-guard/loop_out_final \
GRPO_GUARD_REPLAY_OUT=/root/autodl-tmp/grpo-guard/replay_out_all \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/grpo-guard/.venv/bin/python -m grpo_guard.replay.gradient_probe_torch \
  > /root/autodl-tmp/grpo-guard/replay_all.log 2>&1
' > /dev/null 2>&1 &
echo REPLAY_ALL_LAUNCHED
