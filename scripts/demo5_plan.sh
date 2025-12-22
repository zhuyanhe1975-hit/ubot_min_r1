#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

START_JSON="${1:-examples/demo5_chain_start.json}"
GOAL_JSON="${2:-examples/demo5_cross_goal.json}"
OUT_LOG="${3:-/tmp/demo5_plan.json}"

PLANNER="${PLANNER:-hier}"
MAX_EXPAND="${MAX_EXPAND:-200000}"
MAX_SWAPS="${MAX_SWAPS:-80}"
MAX_INNER_TRIES="${MAX_INNER_TRIES:-300000}"
MAX_INNER_SOLUTIONS="${MAX_INNER_SOLUTIONS:-3}"
IGNORE_PORTS="${IGNORE_PORTS:-0}"

cd "$ROOT_DIR"

extra_args=()
if [[ "$IGNORE_PORTS" == "1" ]]; then
  extra_args+=(--ignore_ports)
fi

exec env/bin/python -m ubot.apps.run_plan \
  --start "$START_JSON" \
  --goal "$GOAL_JSON" \
  --out "$OUT_LOG" \
  --planner "$PLANNER" \
  --max_expand "$MAX_EXPAND" \
  --max_swaps "$MAX_SWAPS" \
  --max_inner_tries "$MAX_INNER_TRIES" \
  --max_inner_solutions "$MAX_INNER_SOLUTIONS" \
  "${extra_args[@]}"

