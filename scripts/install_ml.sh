#!/bin/bash
set -e
cd /root/autodl-tmp/grpo-guard
source .venv/bin/activate
export PATH=/root/uv:$PATH
pip install -q vllm==0.26.0 trl==1.10.0 transformers accelerate peft datasets 2>&1 | tail -5
echo "===VERSIONS==="
python -c "import torch,transformers,trl,vllm,accelerate; print(torch.__version__,transformers.__version__,trl.__version__,vllm.__version__,accelerate.__version__)"
echo "===MODEL==="
HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; p=snapshot_download(\"Qwen/Qwen3-4B\", local_dir=\"/root/autodl-tmp/models/Qwen3-4B\"); print(\"MODEL_AT\", p)"
pip freeze > /root/autodl-tmp/grpo-guard/env_freeze.txt
echo INSTALL_DONE
