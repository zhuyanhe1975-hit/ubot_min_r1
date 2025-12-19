#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

START_JSON="${1:-examples/demo4_chain_start.json}"
GOAL_JSON="${2:-examples/demo4_loop_goal.json}"
OUT_LOG="${3:-/tmp/demo4_plan.json}"
MAX_EXPAND="${MAX_EXPAND:-200000}"

cd "$ROOT_DIR"

exec env/bin/python -m ubot.apps.run_plan \
  --start "$START_JSON" \
  --goal "$GOAL_JSON" \
  --out "$OUT_LOG" \
  --max_expand "$MAX_EXPAND"

