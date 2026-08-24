#!/usr/bin/env bash
# D18 P1-2 seeds study — quick serial runner for the remaining runs
# (on_s1 is already running manually; this does on_s2, off_s0..2).
# Assumes an exclusive GPU window (agent-ttrl idle).  Each run: 10 steps.
set -uo pipefail
REPO=/root/autodl-tmp/grpo-guard/repo
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
LOG=/root/autodl-tmp/grpo-guard/quick_runner.log
export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

echo "$(date) quick runner start" >> $LOG
for spec in "on 2" "off 0" "off 1" "off 2"; do
  set -- $spec
  mode=$1; seed=$2
  OUT=/root/autodl-tmp/grpo-guard/guard_seeds/${mode}_s${seed}
  if [ -f "$OUT/rl_training.json" ]; then
    echo "$(date) skip (done): $mode seed=$seed" >> $LOG
    continue
  fi
  rm -rf $OUT
  echo "$(date) run: $mode seed=$seed" >> $LOG
  FLAGS="--steps 10 --seed $seed"
  [ "$mode" = "off" ] && FLAGS="--guard-off --steps 10 --seed $seed"
  GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
  GRPO_GUARD_LOOP_OUT=$OUT GRPO_GUARD_REPO=$REPO PYTHONPATH=$REPO/src \
  $PY $REPO/examples/countdown/rl_training_loop.py $FLAGS \
    > /root/autodl-tmp/grpo-guard/guard_seeds/${mode}_s${seed}.log 2>&1
  echo "$(date) done: $mode seed=$seed rc=$?" >> $LOG
done
echo "$(date) quick runner DONE" >> $LOG
