from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ubot.core.graph import add_edge, empty_state
from ubot.core.types import Edge, Port, UBotState


_DIRS: List[Tuple[Port, Tuple[int, int], Port]] = [
    ("E", (1, 0), "W"),
    ("W", (-1, 0), "E"),
    ("N", (0, 1), "S"),
    ("S", (0, -1), "N"),
]


@dataclass
class PositionedState:
    state: UBotState
    grid_xy: Dict[int, Tuple[int, int]]


def random_connected_state(
    n_modules: int,
    *,
    ports: Optional[List[Port]] = None,
    seed: Optional[int] = None,
) -> PositionedState:
    """
    Generate a random connected topology on a 2D unit grid (0.1 m per cell in downstream use).
    - All module centers are unique grid cells (no overlap)
    - Graph is connected (tree)
    """
    if n_modules <= 0:
        raise ValueError("n_modules must be positive.")

    rng = random.Random(seed)
    ports = ports or ["N", "S", "E", "W"]

    positions: Dict[int, Tuple[int, int]] = {0: (0, 0)}
    occupied: Set[Tuple[int, int]] = {(0, 0)}
    frontier: Set[Tuple[int, int]] = set()
    joints: Dict[int, List[float]] = {}

    def add_frontier(x: int, y: int) -> None:
        for _p, (dx, dy), _opp in _DIRS:
            cell = (x + dx, y + dy)
            if cell not in occupied:
                frontier.add(cell)

    add_frontier(0, 0)
    edges: List[Edge] = []
    joints[0] = [0.0, 0.0]

    for mid in range(1, n_modules):
        if not frontier:
            raise RuntimeError("Frontier unexpectedly empty; cannot place more modules.")
        x, y = rng.choice(list(frontier))
        frontier.remove((x, y))

        # Find neighbors already placed (guaranteed to exist)
        neighbors = []
        for nid, (px, py) in positions.items():
            dx, dy = x - px, y - py
            if abs(dx) + abs(dy) == 1:
                neighbors.append((nid, dx, dy))
        if not neighbors:
            # This should not happen because frontier is derived from neighbors of occupied cells.
            raise RuntimeError("No neighboring module found for new placement.")

        parent, dx, dy = rng.choice(neighbors)
        if (dx, dy) == (1, 0):
            pu, pv = "E", "W"
        elif (dx, dy) == (-1, 0):
            pu, pv = "W", "E"
        elif (dx, dy) == (0, 1):
            pu, pv = "N", "S"
        else:
            pu, pv = "S", "N"

        edges.append(Edge(u=parent, v=mid, pu=pu, pv=pv))

        positions[mid] = (x, y)
        occupied.add((x, y))
        add_frontier(x, y)
        joints[mid] = [0.0, 0.0]

    s = empty_state(list(range(n_modules)), ports)
    for e in edges:
        add_edge(s, e)
    s.q = joints

    return PositionedState(state=s, grid_xy=positions)
