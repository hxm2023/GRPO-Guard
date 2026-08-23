#!/usr/bin/env bash
# GPU monitor + auto-resume (D18, user: agent-ttrl gets GPU priority).
# NEVER races agent-ttrl: launch our experiments ONLY after (a) no
# agent-ttrl training process, (b) no Mistral vLLM server, and (c) both
# GPUs idle for >= 10 consecutive minutes — i.e. its ctl chain finished.
# Then: resume the interrupted P0-fixed RL training (--resume), and after
# it finishes, run the no-op sync detection experiment.
set -u
LOG=/root/autodl-tmp/grpo-guard/monitor.log
REPO=/root/autodl-tmp/grpo-guard/repo
IDLE_SINCE=""
RL_STARTED=0

echo "$(date) monitor start (agent-ttrl priority: idle >= 600s AND no ttrl procs)" >> $LOG

while true; do
  FREE0=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | awk -F'[, ]+' '/^0/{print $2}')
  FREE1=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | awk -F'[, ]+' '/^1/{print $2}')
  UTIL0=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F'[, ]+' '/^0/{print $2}')
  UTIL1=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -F'[, ]+' '/^1/{print $2}')
  # agent-ttrl busy if any tau2 training or Mistral vLLM process exists.
  # Match the .py / model path so deploy-SCRIPT text (which mentions these
  # strings) is not mistaken for a live process.
  TTRL_BUSY=$(ps aux | grep -E '[t]au2_agent_stream\.py|[M]istral-7B-Instruct' | wc -l)
  NOW=$(date +%s)

  IDLE=0
  if [ -n "$FREE0" ] && [ -n "$FREE1" ] && [ "$FREE0" -gt 60000 ] && [ "$FREE1" -gt 60000 ] \
     && [ "${UTIL0:-100}" -lt 10 ] && [ "${UTIL1:-100}" -lt 10 ] && [ "$TTRL_BUSY" = "0" ]; then
    IDLE=1
  fi

  if [ "$IDLE" = "1" ]; then
    if [ -z "$IDLE_SINCE" ]; then
      IDLE_SINCE=$NOW
      echo "$(date) agent-ttrl finished + GPU idle begins (free0=$FREE0 free1=$FREE1)" >> $LOG
    fi
    if [ $((NOW - IDLE_SINCE)) -ge 600 ] && [ "$RL_STARTED" = "0" ]; then
      echo "$(date) idle 600s, no ttrl procs — launching RL training --resume" >> $LOG
      bash $REPO/scripts/launch_rl_resume.sh
      RL_STARTED=1
      while ps aux | grep -q '[r]l_training_loop'; do sleep 120; done
      echo "$(date) RL resume finished — launching no-op sync experiment" >> $LOG
      bash $REPO/scripts/launch_noop.sh
      echo "$(date) no-op launched; monitor done" >> $LOG
      exit 0
    fi
  else
    if [ -n "$IDLE_SINCE" ]; then
      echo "$(date) busy again (ttrl=$TTRL_BUSY free0=$FREE0 free1=$FREE1 util0=${UTIL0} util1=${UTIL1})" >> $LOG
    fi
    IDLE_SINCE=""
  fi
  sleep 60
done
