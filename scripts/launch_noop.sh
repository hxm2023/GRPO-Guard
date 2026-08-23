#!/usr/bin/env bash
# Launch the no-op sync detection experiment (P0-2): server-vs-trainer
# greedy sketch exposes a silent no-op update_named_param — the original
# static-rollout accident.  GPU1-only (agent-ttrl may still hold GPU0).
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_OUT=/root/autodl-tmp/grpo-guard/sync_noop_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
CUDA_VISIBLE_DEVICES=1 \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/sync_noop_experiment.py \
  > /root/autodl-tmp/grpo-guard/sync_noop.log 2>&1
' > /dev/null 2>&1 &
echo NOOP_LAUNCHED
