#!/usr/bin/env bash
# P1-2 3-seed guard on/off study (D18): 6 runs serially on autodl2 —
# guard-on x seeds 0/1/2 + guard-off x seeds 0/1/2, 10 steps each.
# Each run gets its own OUT_DIR + log; results under
# /root/autodl-tmp/grpo-guard/guard_seeds/{on,off}_s{seed}/rl_training.json
set -u
REPO=/root/autodl-tmp/grpo-guard/repo
PY=/root/autodl-tmp/grpo-guard/.venv/bin/python
LOG=/root/autodl-tmp/grpo-guard/seeds_runner.log

echo "$(date) seeds runner start (6 runs x 10 steps)" >> $LOG
for mode in on off; do
  for seed in 0 1 2; do
    OUT=/root/autodl-tmp/grpo-guard/guard_seeds/${mode}_s${seed}
    rm -rf $OUT
    echo "$(date) run: $mode seed=$seed" >> $LOG
    cd $REPO
    export LD_LIBRARY_PATH=/root/autodl-tmp/grpo-guard/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
    if [ "$mode" = "off" ]; then
      GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
      GRPO_GUARD_LOOP_OUT=$OUT \
      GRPO_GUARD_REPO=$REPO \
      PYTHONPATH=$REPO/src \
      $PY examples/countdown/rl_training_loop.py --guard-off --steps 10 --seed $seed \
        > /root/autodl-tmp/grpo-guard/guard_seeds/${mode}_s${seed}.log 2>&1
    else
      GRPO_GUARD_MODEL_PATH=/root/autodl-tmp/models/Qwen3-4B \
      GRPO_GUARD_LOOP_OUT=$OUT \
      GRPO_GUARD_REPO=$REPO \
      PYTHONPATH=$REPO/src \
      $PY examples/countdown/rl_training_loop.py --steps 10 --seed $seed \
        > /root/autodl-tmp/grpo-guard/guard_seeds/${mode}_s${seed}.log 2>&1
    fi
    echo "$(date) done: $mode seed=$seed rc=$?" >> $LOG
  done
done
echo "$(date) seeds runner DONE" >> $LOG
