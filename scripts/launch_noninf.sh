#!/usr/bin/env bash
# E2: guard on/off non-inferiority (official TRL path, 5 seeds x 2 arms).
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
export ARM="${GRPO_GUARD_ARM:-on}"
export SEED="${GRPO_GUARD_SEED:-20260825}"
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_OUT=/root/autodl-tmp/grpo-guard/noninf_out_${ARM}_${SEED} \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
GRPO_GUARD_STEPS=${GRPO_GUARD_STEPS:-30} \
GRPO_GUARD_ARM=${ARM} \
GRPO_GUARD_SEED=${SEED} \
GRPO_GUARD_VLLM_MEM=${GRPO_GUARD_VLLM_MEM:-0.25} \
GRPO_GUARD_SERVER_DEVICE=${GRPO_GUARD_SERVER_DEVICE:-1} \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/non_inferiority.py \
  > /root/autodl-tmp/grpo-guard/noninf_${ARM}.log 2>&1
' > /dev/null 2>&1 &
echo NONINF_LAUNCHED_${ARM}
