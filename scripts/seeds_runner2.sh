#!/usr/bin/env bash
# P1-2 3-seed study v2 (robust): serial 6 runs with strict error handling.
set -uo pipefail
REPO=/root/autodl-tmp/grpo-guard/repo
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
LOG=/root/autodl-tmp/grpo-guard/seeds_runner.log
export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

echo "$(date) seeds runner v2 start" >> $LOG
for mode in on off; do
  for seed in 0 1 2; do
    OUT=/root/autodl-tmp/grpo-guard/guard_seeds/${mode}_s${seed}
    mkdir -p /root/autodl-tmp/grpo-guard/guard_seeds
    if [ -f "$OUT/rl_training.json" ]; then
      echo "$(date) skip (done): $mode seed=$seed" >> $LOG
      continue
    fi
    rm -rf $OUT
    echo "$(date) run: $mode seed=$seed" >> $LOG
    if [ "$mode" = "off" ]; then
      GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
      GRPO_GUARD_LOOP_OUT=$OUT GRPO_GUARD_REPO=$REPO PYTHONPATH=$REPO/src \
      $PY $REPO/examples/countdown/rl_training_loop.py --guard-off --steps 10 --seed $seed \
        > /root/autodl-tmp/grpo-guard/guard_seeds/${mode}_s${seed}.log 2>&1
    else
      GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
      GRPO_GUARD_LOOP_OUT=$OUT GRPO_GUARD_REPO=$REPO PYTHONPATH=$REPO/src \
      $PY $REPO/examples/countdown/rl_training_loop.py --steps 10 --seed $seed \
        > /root/autodl-tmp/grpo-guard/guard_seeds/${mode}_s${seed}.log 2>&1
    fi
    RC=$?
    echo "$(date) done: $mode seed=$seed rc=$RC" >> $LOG
  done
done
echo "$(date) seeds runner v2 DONE" >> $LOG
