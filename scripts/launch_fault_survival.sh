#!/usr/bin/env bash
# E1: fault-injected training survival (official TRL path, guard on/off).
# Env: GRPO_GUARD_ARM=on|off, GRPO_GUARD_SEED, GRPO_GUARD_STEPS,
#      GRPO_GUARD_FAULT_STEPS, GRPO_GUARD_FAULT_KINDS, GRPO_GUARD_VLLM_MEM.
set -euo pipefail
cd /root/autodl-tmp/grpo-guard/repo
ARM="${GRPO_GUARD_ARM:-on}"
SEED="${GRPO_GUARD_SEED:-20260825}"
nohup bash -c '
cd /root/autodl-tmp/grpo-guard/repo
export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
GRPO_GUARD_OUT=/root/autodl-tmp/grpo-guard/fault_survival_out \
GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
GRPO_GUARD_STEPS=${GRPO_GUARD_STEPS:-30} \
GRPO_GUARD_ARM='${ARM}' \
GRPO_GUARD_SEED='${SEED}' \
GRPO_GUARD_FAULT_STEPS=${GRPO_GUARD_FAULT_STEPS:-10,20} \
GRPO_GUARD_FAULT_KINDS=${GRPO_GUARD_FAULT_KINDS:-F3,F2} \
GRPO_GUARD_VLLM_MEM=${GRPO_GUARD_VLLM_MEM:-0.25} \
GRPO_GUARD_SERVER_DEVICE=${GRPO_GUARD_SERVER_DEVICE:-0} \
PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/fault_survival.py \
  > /root/autodl-tmp/grpo-guard/fault_survival_'${ARM}'.log 2>&1
' > /dev/null 2>&1 &
echo FAULT_SURVIVAL_LAUNCHED_${ARM}
