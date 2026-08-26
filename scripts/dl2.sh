#!/bin/bash
cd /root/autodl-tmp/grpo-guard
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
nohup python -u -c "
import os, sys
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from huggingface_hub import snapshot_download
print('starting download...', flush=True)
p = snapshot_download('Qwen/Qwen3-4B', local_dir='/root/autodl-tmp/models/Qwen3-4B')
print('MODEL_AT', p, flush=True)
" > dl2.log 2>&1 &
echo LAUNCHED_PID $!
