from __future__ import annotations

import argparse
import json
import os

from ubot.core.graph import edges_undirected
from ubot.core.random_state import random_connected_state


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a random connected UBot topology on a 0.1 m grid.")
    ap.add_argument("--n", type=int, required=True, help="Number of modules")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    ap.add_argument(
        "--out",
        default="random_state.json",
        help="Output JSON path (default: random_state.json in CWD)",
    )
    args = ap.parse_args()

    result = random_connected_state(args.n, seed=args.seed)
    out_path = os.path.abspath(args.out)
    payload = {
        "modules": sorted(list(result.state.adj.keys())),
        "ports": list(result.state.ports),
        "edges": [e.__dict__ for e in edges_undirected(result.state)],
        "joints": {int(k): list(v) for k, v in (result.state.q or {}).items()},
        "meta": {"grid_xy": result.grid_xy, "grid_spacing_m": 0.1},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[ok] Wrote random connected state with {args.n} modules to {out_path}")


if __name__ == "__main__":
    main()
