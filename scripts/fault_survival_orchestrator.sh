#!/usr/bin/env bash
# E1 orchestrator (server-side): runs on/off x seed arms sequentially,
# waits for a dual-GPU window, retries on server-start failure.
set -u
cd /root/autodl-tmp/grpo-guard/repo

# clean leftovers from previous attempts (Guard's own venv only)
ps aux | grep -E 'grpo-guard/.venv' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 5

for arm_seed in "on 20260825" "on 20260826" "off 20260825" "off 20260826"; do
  ARM=${arm_seed% *}
  SEED=${arm_seed#* }
  for attempt in 1 2 3; do
    echo "=== $ARM s$SEED attempt $attempt: waiting for GPU window ==="
    ok=0
    for i in $(seq 1 40); do
      sleep 120
      U0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed -n 1p | tr -d ' MiB')
      U1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | sed -n 2p | tr -d ' MiB')
      F0=$((84000 - U0)); F1=$((84000 - U1))
      echo "  free gpu0=${F0}MiB gpu1=${F1}MiB"
      if [ "$F0" -gt 58000 ] && [ "$F1" -gt 26000 ]; then ok=1; break; fi
    done
    [ "$ok" = 1 ] || { echo "NO_WINDOW"; break; }
    rm -rf /root/autodl-tmp/grpo-guard/fault_survival_out
    GRPO_GUARD_ARM=$ARM GRPO_GUARD_SEED=$SEED GRPO_GUARD_VLLM_MEM=0.3 \
      bash scripts/launch_fault_survival.sh
    echo "=== launched $ARM s$SEED; waiting for result ==="
    done_flag=0
    for j in $(seq 1 60); do
      sleep 30
      if [ -f /root/autodl-tmp/grpo-guard/fault_survival_out/fault_survival.json ]; then
        done_flag=1; break
      fi
      if grep -q "Traceback" /root/autodl-tmp/grpo-guard/fault_survival_${ARM}.log 2>/dev/null; then
        break
      fi
    done
    if [ "$done_flag" = 1 ]; then
      echo "=== DONE $ARM s$SEED ==="
      break
    fi
    echo "=== $ARM s$SEED attempt $attempt FAILED; retrying ==="
    ps aux | grep -E 'grpo-guard/.venv' | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    sleep 30
  done
done
echo ALL_ARMS_FINISHED
# marker-test
