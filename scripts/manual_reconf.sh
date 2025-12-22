#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <state.json> [template.xml]"
  echo "Example: $0 examples/demo7_line_start.json models/ubot/scene.xml"
  exit 2
fi

STATE_JSON="$1"
TEMPLATE_XML="${2:-models/ubot/scene.xml}"

STEP_DT="${STEP_DT:-1.0}"
INTERP_STEPS="${INTERP_STEPS:-120}"

cd "$ROOT_DIR"

exec env/bin/python -m ubot.apps.show_mujoco \
  --state "$STATE_JSON" \
  --template "$TEMPLATE_XML" \
  --interactive \
  --step_dt "$STEP_DT" \
  --interp_steps "$INTERP_STEPS"

