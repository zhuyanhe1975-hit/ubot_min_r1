from __future__ import annotations
import argparse
import time
from ubot.core.io import load_state_json
from ubot.core.planner import plan_astar
from ubot.core.validator import validate_action
from ubot.adapters.sim_adapter import SimAdapter
from ubot.core.queue import save_log

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Path to start state JSON")
    ap.add_argument("--goal", required=True, help="Path to goal state JSON")
    ap.add_argument("--out", default="ubot_run_log.json", help="Output log JSON")
    ap.add_argument("--max_expand", type=int, default=50000)
    args = ap.parse_args()

    start = load_state_json(args.start)
    goal = load_state_json(args.goal)

    t0 = time.time()
    plan = plan_astar(start, goal, max_expand=args.max_expand)
    dt = time.time() - t0
    if plan is None:
        raise SystemExit(f"No plan found within max_expand={args.max_expand}")

    print(f"Found plan with {len(plan)} actions in {dt:.3f}s")
    for i, a in enumerate(plan):
        print(f"{i:03d}: {a}")

    adapter = SimAdapter(start)
    executed = []
    for a in plan:
        s = adapter.get_state()
        validate_action(s, a)
        adapter.apply(a)
        executed.append(a)

    save_log(args.out, plan=plan, executed=executed, meta={"planner": "astar", "runtime_sec": dt})
    print(f"Wrote log to: {args.out}")

if __name__ == "__main__":
    main()
