from __future__ import annotations
import argparse
import time
from ubot.core.io import load_state_json
from ubot.core.planner import plan_astar, plan_hierarchical_edge_swap
from ubot.core.validator import validate_action
from ubot.adapters.sim_adapter import SimAdapter
from ubot.core.queue import save_log

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Path to start state JSON")
    ap.add_argument("--goal", required=True, help="Path to goal state JSON")
    ap.add_argument("--out", default="ubot_run_log.json", help="Output log JSON")
    ap.add_argument("--max_expand", type=int, default=50000)
    ap.add_argument("--planner", choices=["astar", "hier"], default="astar")
    ap.add_argument("--max_swaps", type=int, default=50, help="(hier) max edge-swap steps")
    ap.add_argument("--max_inner_tries", type=int, default=300000, help="(hier) max joint combos per attach attempt")
    ap.add_argument("--max_inner_solutions", type=int, default=3, help="(hier) max joint solutions per attach attempt")
    ap.add_argument("--ignore_ports", action="store_true", help="(hier) match goal topology ignoring port names")
    ap.add_argument("--max_port_pairs", type=int, default=4, help="(hier) max port-pair tries per missing edge when ignoring ports")
    args = ap.parse_args()

    start = load_state_json(args.start)
    goal = load_state_json(args.goal)

    t0 = time.time()
    if args.planner == "hier":
        plan = plan_hierarchical_edge_swap(
            start,
            goal,
            max_swaps=args.max_swaps,
            max_inner_tries=args.max_inner_tries,
            max_inner_solutions=args.max_inner_solutions,
            ignore_ports=args.ignore_ports,
            max_port_pairs=args.max_port_pairs,
        )
    else:
        plan = plan_astar(start, goal, max_expand=args.max_expand)
    dt = time.time() - t0
    if plan is None:
        if args.planner == "hier":
            raise SystemExit("No plan found with hierarchical edge-swap planner.")
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

    save_log(args.out, plan=plan, executed=executed, meta={"planner": args.planner, "runtime_sec": dt})
    print(f"Wrote log to: {args.out}")

if __name__ == "__main__":
    main()
