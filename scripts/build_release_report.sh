#!/usr/bin/env bash
# Day 5 release report bundle (design doc §19.2).
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo uncommitted)"
uv run grpo-guard report --artifact-dir artifacts/v0.1.0 --commit "${COMMIT}"
echo "RELEASE REPORT BUNDLE COMPLETE"
