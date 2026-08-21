#!/usr/bin/env bash
# Day 3 reason-coded fault matrix (design doc §16.2 Correctness Gate inputs).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run grpo-guard fault-matrix \
  --config configs/faults/f1_f4_v01.yaml \
  --guard-mode strict_on_policy \
  --out artifacts/v0.1.0 \
  --freeze tests/frozen/f1_f4_v01
echo "FAULT MATRIX RUN COMPLETE"
