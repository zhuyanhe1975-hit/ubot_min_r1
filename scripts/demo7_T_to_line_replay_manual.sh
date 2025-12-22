#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STATE_JSON="${1:-examples/demo7_T_to_line_start.json}"
LOG_JSON="${2:-examples/demo7_T_to_line_manual_log.json}"
TEMPLATE_XML="${3:-models/ubot/scene.xml}"

STEP_DT="${STEP_DT:-1.0}"
INTERP_STEPS="${INTERP_STEPS:-120}"

cd "$ROOT_DIR"

exec env/bin/python -m ubot.apps.show_mujoco \
  --state "$STATE_JSON" \
  --template "$TEMPLATE_XML" \
  --log "$LOG_JSON" \
  --step_dt "$STEP_DT" \
  --interp_steps "$INTERP_STEPS"

