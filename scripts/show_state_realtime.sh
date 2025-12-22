#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <state.json> [template.xml]"
  echo "Env:"
  echo "  POLL_SEC=0.2   Poll interval (seconds)"
  echo "Example:"
  echo "  $0 examples/demo7_line_start.json models/ubot/scene.xml"
  exit 2
fi

STATE_JSON="$1"
TEMPLATE_XML="${2:-models/ubot/scene.xml}"
POLL_SEC="${POLL_SEC:-0.2}"

cd "$ROOT_DIR"

if [[ ! -f "$STATE_JSON" ]]; then
  echo "State file not found: $STATE_JSON" >&2
  exit 2
fi

exec env/bin/python -m ubot.apps.show_mujoco \
  --state "$STATE_JSON" \
  --template "$TEMPLATE_XML" \
  --watch_state "$POLL_SEC"
