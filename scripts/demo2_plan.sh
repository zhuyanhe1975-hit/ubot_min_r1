#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

START_JSON="${1:-examples/demo2_connected_start.json}"
GOAL_JSON="${2:-examples/demo2_connected_goal.json}"
OUT_LOG="${3:-/tmp/demo2_plan.json}"

PLANNER="${PLANNER:-astar}"
MAX_EXPAND="${MAX_EXPAND:-50000}"
MAX_SWAPS="${MAX_SWAPS:-50}"
MAX_INNER_TRIES="${MAX_INNER_TRIES:-300000}"
MAX_INNER_SOLUTIONS="${MAX_INNER_SOLUTIONS:-3}"

cd "$ROOT_DIR"

exec env/bin/python -m ubot.apps.run_plan \
  --start "$START_JSON" \
  --goal "$GOAL_JSON" \
  --out "$OUT_LOG" \
  --planner "$PLANNER" \
  --max_expand "$MAX_EXPAND" \
  --max_swaps "$MAX_SWAPS" \
  --max_inner_tries "$MAX_INNER_TRIES" \
  --max_inner_solutions "$MAX_INNER_SOLUTIONS"

