#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_JSON="${1:-examples/demo7_T_to_line_start.json}"
GOAL_JSON="${2:-examples/demo7_T_to_line_goal.json}"
LOG_JSON="${3:-examples/demo7_T_to_line_manual_log.json}"

cd "$ROOT_DIR"

exec env/bin/python - "$START_JSON" "$GOAL_JSON" "$LOG_JSON" <<'PY'
import json
import sys
from ubot.core.io import load_state_json
from ubot.adapters.sim_adapter import SimAdapter
from ubot.core.types import Action
from ubot.core.graph import edges_undirected

start_path, goal_path, log_path = sys.argv[1:4]
start = load_state_json(start_path)
goal = load_state_json(goal_path)

with open(log_path, "r", encoding="utf-8") as f:
    log = json.load(f)

actions = []
for a in log.get("plan", []):
    actions.append(
        Action(
            kind=a["kind"],
            u=int(a["u"]),
            v=int(a["v"]),
            pu=a.get("pu"),
            pv=a.get("pv"),
            twist=a.get("twist"),
            j1=a.get("j1"),
            j2=a.get("j2"),
        )
    )

ad = SimAdapter(start)
for i, a in enumerate(actions):
    try:
        ad.apply(a)
    except Exception as e:
        raise SystemExit(f"FAILED at step {i}: {a} -> {e}")

end = ad.get_state()

def edge_sig(st):
    return sorted((e.u, e.v, e.pu, e.pv, int(e.twist) % 360) for e in edges_undirected(st))

ok_edges = edge_sig(end) == edge_sig(goal)
ok_joints = end.q == goal.q

print("OK edges:", ok_edges)
print("OK joints:", ok_joints)
if not ok_edges:
    print("end edges:", edge_sig(end))
    print("goal edges:", edge_sig(goal))
if not ok_joints:
    print("end q:", end.q)
    print("goal q:", goal.q)
raise SystemExit(0 if (ok_edges and ok_joints) else 1)
PY
