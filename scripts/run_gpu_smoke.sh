#!/usr/bin/env bash
# Official TRL+vLLM server-mode smoke on autodl2 (design doc §4.1.1).
# Layout: GPU1 = trl vllm-serve; GPU0 = GRPOTrainer (server mode).
set -euo pipefail

REPO_DIR="${1:-/root/autodl-tmp/grpo-guard/repo}"
OUT_DIR="${2:-/root/autodl-tmp/grpo-guard/smoke_out}"
MODEL="${GRPO_GUARD_MODEL:-Qwen/Qwen3-4B}"
VLLM_PORT="${VLLM_PORT:-8000}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

cd "${REPO_DIR}"
source /root/autodl-tmp/grpo-guard/.venv/bin/activate

echo "[smoke] starting vLLM server on GPU1 (${MODEL})"
CUDA_VISIBLE_DEVICES=1 nohup trl vllm-serve \
  --model "${MODEL}" \
  --port "${VLLM_PORT}" \
  --gpu-memory-utilization 0.6 \
  --max-model-len 2048 \
  > "${LOG_DIR}/vllm_server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[smoke] waiting for server health..."
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "[smoke] server healthy after ${i}s"
    break
  fi
  sleep 2
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[smoke] server process died"
    tail -30 "${LOG_DIR}/vllm_server.log"
    exit 1
  fi
done

echo "[smoke] running GRPOTrainer on GPU0 (server mode, 1 committed step)"
GRPO_GUARD_MODEL="${MODEL}" \
GRPO_GUARD_VLLM_PORT="${VLLM_PORT}" \
GRPO_GUARD_SMOKE_OUT="${OUT_DIR}" \
CUDA_VISIBLE_DEVICES=0 python examples/countdown/smoke_train.py \
  > "${LOG_DIR}/trainer.log" 2>&1
TRAINER_RC=$?

echo "[smoke] trainer exit code: ${TRAINER_RC}"
if [ "${TRAINER_RC}" -ne 0 ]; then
  tail -40 "${LOG_DIR}/trainer.log"
  exit "${TRAINER_RC}"
fi

python - <<PY
import json, sys
from pathlib import Path
out = Path("${OUT_DIR}/smoke_result.json")
r = json.loads(out.read_text())
ok = r.get("committed_optimizer_steps") == 1 and len(r.get("trl_observed_sync_calls", [])) > 0
print(f"[smoke] committed_steps={r.get('committed_optimizer_steps')} sync_calls={len(r.get('trl_observed_sync_calls', []))}")
sys.exit(0 if ok else 1)
PY
echo "[smoke] COMPATIBILITY GATE SMOKE PASSED"
