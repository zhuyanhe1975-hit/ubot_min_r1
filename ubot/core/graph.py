from __future__ import annotations
from typing import Dict, List
from ubot.core.types import UBotState, Edge, Port

class GraphError(Exception):
    pass

def empty_state(modules: List[int], ports: List[Port]) -> UBotState:
    return UBotState(adj={m: {} for m in modules}, ports=list(ports))

def add_edge(s: UBotState, e: Edge) -> None:
    if e.u == e.v:
        raise GraphError("Self-edge not allowed.")
    if e.pu in s.adj.get(e.u, {}):
        raise GraphError(f"Port occupied: {e.u}.{e.pu}")
    if e.pv in s.adj.get(e.v, {}):
        raise GraphError(f"Port occupied: {e.v}.{e.pv}")
    s.adj.setdefault(e.u, {})[e.pu] = (e.v, e.pv)
    s.adj.setdefault(e.v, {})[e.pv] = (e.u, e.pu)

def remove_edge(s: UBotState, u: int, v: int) -> None:
    ports_u = list(s.adj.get(u, {}).items())
    for pu, (vv, pv) in ports_u:
        if vv == v:
            del s.adj[u][pu]
            if pv in s.adj.get(v, {}) and s.adj[v][pv][0] == u:
                del s.adj[v][pv]
            return
    raise GraphError(f"No edge between {u} and {v}")

def edges_undirected(s: UBotState) -> List[Edge]:
    out: List[Edge] = []
    seen = set()
    for u, ports in s.adj.items():
        for pu, (v, pv) in ports.items():
            if (u, v, pu, pv) in seen or (v, u, pv, pu) in seen:
                continue
            if u < v:
                out.append(Edge(u=u, v=v, pu=pu, pv=pv))
                seen.add((u, v, pu, pv))
                seen.add((v, u, pv, pu))
            else:
                out.append(Edge(u=v, v=u, pu=pv, pv=pu))
                seen.add((v, u, pv, pu))
                seen.add((u, v, pu, pv))
    out.sort(key=lambda e: (e.u, e.v, e.pu, e.pv))
    return out

def clone_state(s: UBotState) -> UBotState:
    return UBotState(
        adj={u: dict(ports) for u, ports in s.adj.items()},
        ports=list(s.ports),
        q=None if s.q is None else {mid: list(vals) for mid, vals in s.q.items()},
        parent=None if s.parent is None else {mid: p for mid, p in s.parent.items()},
        pose=None
        if s.pose is None
        else {mid: tuple(float(v) for v in vals) for mid, vals in s.pose.items()},
        weld=None
        if s.weld is None
        else {(int(u), str(pu), int(v), str(pv)): tuple(float(x) for x in tf) for (u, pu, v, pv), tf in s.weld.items()},
    )


def compute_parent_map(s: UBotState, root: int = 0) -> Dict[int, Optional[int]]:
    """
    Compute a BFS tree rooted at `root`. Unreachable modules get parent=None.
    """
    parents: Dict[int, Optional[int]] = {m: None for m in s.adj.keys()}
    if root not in s.adj:
        return parents

    frontier = [root]
    visited = {root}
    parents[root] = None

    while frontier:
        u = frontier.pop()
        for _pu, (v, _pv) in s.adj.get(u, {}).items():
            if v in visited:
                continue
            visited.add(v)
            parents[v] = u
            frontier.append(v)

    return parents


def init_pose_zero(s: UBotState) -> Dict[int, Tuple[float, float, float, float]]:
    """Backward-compat placeholder (deprecated)."""
    return {m: (0.0, 0.0, 0.0, 0.0) for m in s.adj.keys()}

def free_ports(s: UBotState, u: int) -> List[Port]:
    used = set(s.adj.get(u, {}).keys())
    return [p for p in s.ports if p not in used]
