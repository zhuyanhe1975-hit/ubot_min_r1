from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ubot.core.graph import edges_undirected
from ubot.core.types import Edge, Port, UBotState


_DIR: Dict[Port, Tuple[int, int]] = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
_OPP: Dict[Port, Port] = {"N": "S", "S": "N", "E": "W", "W": "E"}


@dataclass(frozen=True)
class LayoutResult:
    grid_xy: Dict[int, Tuple[int, int]]
    warnings: List[str]


def _iter_constraints(edges: Iterable[Edge]) -> Iterable[Tuple[int, int, Tuple[int, int], Optional[str]]]:
    """
    Yields directed constraints of the form:
      v = u + delta
    and optionally a warning string when port pairing looks inconsistent.
    """
    for e in edges:
        du = _DIR.get(e.pu)
        dv = _DIR.get(e.pv)

        warn = None
        if du is None or dv is None:
            warn = f"Unknown port name in edge ({e.u},{e.v}) pu={e.pu} pv={e.pv}; layout may be wrong."
        else:
            expected_pv = _OPP[e.pu]
            expected_pu = _OPP[e.pv]
            if e.pv != expected_pv and e.pu != expected_pu:
                warn = f"Port pairing mismatch for edge ({e.u},{e.v}) pu={e.pu} pv={e.pv}; expected {expected_pv}/{expected_pu}."

        if du is not None:
            yield (e.u, e.v, du, warn)
            yield (e.v, e.u, (-du[0], -du[1]), warn)


def layout_from_ports(state: UBotState) -> LayoutResult:
    """
    Computes a simple 2D grid layout from port-direction constraints.

    Port convention (typical):
      u.E -- v.W  => v is east of u
      u.N -- v.S  => v is north of u

    Disconnected components are laid out with a fixed separation.
    """
    modules = list(state.adj.keys())
    if not modules:
        return LayoutResult(grid_xy={}, warnings=[])

    warnings: List[str] = []
    grid_xy: Dict[int, Tuple[int, int]] = {}

    edges = edges_undirected(state)
    constraints = list(_iter_constraints(edges))

    outgoing: Dict[int, List[Tuple[int, Tuple[int, int], Optional[str]]]] = {m: [] for m in modules}
    for u, v, delta, warn in constraints:
        outgoing.setdefault(u, []).append((v, delta, warn))

    # Lay out each connected component with an offset to avoid overlap.
    component_idx = 0
    component_sep = 3

    for seed in modules:
        if seed in grid_xy:
            continue

        grid_xy[seed] = (component_idx * component_sep, 0)
        component_idx += 1

        q = deque([seed])
        while q:
            u = q.popleft()
            ux, uy = grid_xy[u]
            for v, (dx, dy), warn in outgoing.get(u, []):
                if warn is not None:
                    warnings.append(warn)

                expected = (ux + dx, uy + dy)
                if v not in grid_xy:
                    grid_xy[v] = expected
                    q.append(v)
                elif grid_xy[v] != expected:
                    warnings.append(
                        f"Layout conflict: {v} expected at {expected} from {u}, but already at {grid_xy[v]}."
                    )

    return LayoutResult(grid_xy=grid_xy, warnings=warnings)

