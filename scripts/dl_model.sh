#!/bin/bash
cd /root/autodl-tmp/grpo-guard
source .venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
python -c "from huggingface_hub import snapshot_download; p=snapshot_download('Qwen/Qwen3-4B', local_dir='/root/autodl-tmp/models/Qwen3-4B'); print('MODEL_AT', p)"
echo MODEL_DL_DONE
