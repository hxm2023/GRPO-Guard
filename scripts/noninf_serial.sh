#!/usr/bin/env bash
# E2 manual serial runner with fast-fail detection (process death -> retry).
set -u
cd /root/autodl-tmp/grpo-guard/repo

for arm_seed in "on 20260828" "on 20260829" "off 20260825" "off 20260826" "off 20260827" "off 20260828" "off 20260829"; do
  ARM=${arm_seed% *}
  SEED=${arm_seed#* }
  for attempt in 1 2 3 4 5; do
    echo "=== $ARM s$SEED attempt $attempt ==="
    ok=0
    for i in $(seq 1 40); do
      sleep 60
      U0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed -n 1p | tr -d ' MiB')
      U1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed -n 2p | tr -d ' MiB')
      F0=$((84000 - U0)); F1=$((84000 - U1))
      if [ "$F0" -gt 55000 ] && [ "$F1" -gt 20000 ]; then ok=1; break; fi
    done
    [ "$ok" = 1 ] || { echo "NO_WINDOW"; break; }
    rm -rf /root/autodl-tmp/grpo-guard/noninf_out_${ARM}_${SEED}
    GRPO_GUARD_ARM=$ARM GRPO_GUARD_SEED=$SEED GRPO_GUARD_VLLM_MEM=0.2 bash scripts/launch_noninf.sh
    done_flag=0
    for j in $(seq 1 40); do
      sleep 30
      if [ -f /root/autodl-tmp/grpo-guard/noninf_out_${ARM}_${SEED}/non_inferiority.json ]; then
        done_flag=1; break
      fi
      if ! ps aux | grep "non_inferiority.py" | grep -v grep > /dev/null; then
        # process died without producing the result -> fail fast
        break
      fi
    done
    if [ "$done_flag" = 1 ]; then
      echo "=== TRAIN DONE $ARM s$SEED ==="
      CKPT=$(ls -d /root/autodl-tmp/grpo-guard/noninf_out_${ARM}_${SEED}/ckpt/checkpoint-* 2>/dev/null | tail -1)
      GRPO_GUARD_CKPT=$CKPT \
      GRPO_GUARD_OUT=/root/autodl-tmp/grpo-guard/noninf_out_${ARM}_${SEED} \
      GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
      GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
      PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
      CUDA_VISIBLE_DEVICES=0 \
      /root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/eval_heldout.py \
        >> /root/autodl-tmp/grpo-guard/noninf_${ARM}.log 2>&1
      echo "=== EVAL DONE $ARM s$SEED ==="
      break
    fi
    echo "=== $ARM s$SEED attempt $attempt FAILED; retrying ==="
    ps aux | grep -E 'grpo-guard/.venv' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    sleep 20
  done
done
echo ALL_REMAINING_FINISHED
