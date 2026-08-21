#!/usr/bin/env bash
# Day 1 CPU contract: unit + contract + property tests (design doc §15).
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra test
uv run pytest tests/unit tests/contract tests/property
echo "CPU CONTRACT TESTS PASSED"
