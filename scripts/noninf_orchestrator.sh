#!/usr/bin/env bash
# E2 orchestrator (server-side): 5 seeds x {on, off} sequentially.
set -u
cd /root/autodl-tmp/grpo-guard/repo
ps aux | grep -E 'grpo-guard/.venv' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 5

for arm in on off; do
  for seed in 20260825 20260826 20260827 20260828 20260829; do
    for attempt in 1 2 3; do
      echo "=== $arm s$seed attempt $attempt: waiting for GPU window ==="
      ok=0
      for i in $(seq 1 40); do
        sleep 120
        U0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed -n 1p | tr -d ' MiB')
        U1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed -n 2p | tr -d ' MiB')
        F0=$((84000 - U0)); F1=$((84000 - U1))
        echo "  free gpu0=${F0}MiB gpu1=${F1}MiB"
        if [ "$F0" -gt 45000 ] && [ "$F1" -gt 24000 ]; then ok=1; break; fi
      done
      [ "$ok" = 1 ] || { echo "NO_WINDOW"; break; }
      rm -rf /root/autodl-tmp/grpo-guard/noninf_out_${arm}_${seed}
      GRPO_GUARD_ARM=$arm GRPO_GUARD_SEED=$seed GRPO_GUARD_VLLM_MEM=0.2 bash scripts/launch_noninf.sh
      echo "=== launched $arm s$seed; waiting ==="
      done_flag=0
      for j in $(seq 1 60); do
        sleep 30
        if [ -f /root/autodl-tmp/grpo-guard/noninf_out_${arm}_${seed}/non_inferiority.json ]; then
          done_flag=1; break
        fi
        if grep -q "Traceback" /root/autodl-tmp/grpo-guard/noninf_${arm}.log 2>/dev/null; then break; fi
      done
      if [ "$done_flag" = 1 ]; then
        echo "=== DONE $arm s$seed (training); running held-out eval ==="
        CKPT=$(ls -d /root/autodl-tmp/grpo-guard/noninf_out_${arm}_${seed}/ckpt/checkpoint-* 2>/dev/null | tail -1)
        GRPO_GUARD_CKPT=$CKPT \
        GRPO_GUARD_OUT=/root/autodl-tmp/grpo-guard/noninf_out_${arm}_${seed} \
        GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
        GRPO_GUARD_REPO=/root/autodl-tmp/grpo-guard/repo \
        PYTHONPATH=/root/autodl-tmp/grpo-guard/repo/src \
        CUDA_VISIBLE_DEVICES=0 \
        /root/autodl-tmp/grpo-guard/.venv/bin/python examples/countdown/eval_heldout.py \
          >> /root/autodl-tmp/grpo-guard/noninf_${arm}.log 2>&1
        echo "=== EVAL DONE $arm s$seed ==="
        break
      fi
      echo "=== $arm s$seed attempt $attempt FAILED; retrying ==="
      ps aux | grep -E 'grpo-guard/.venv' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
      sleep 30
    done
  done
done
echo ALL_NONINF_FINISHED
