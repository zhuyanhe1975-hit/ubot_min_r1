from __future__ import annotations
from dataclasses import dataclass
from heapq import heappush, heappop
from itertools import count
from typing import Dict, List, Optional, Set, Tuple

from ubot.core.types import UBotState, Action, Edge
from ubot.core.graph import clone_state, edges_undirected, add_edge, remove_edge, free_ports
from ubot.core.validator import validate_action, ValidationError

def state_sig(s: UBotState) -> Tuple[Tuple[int, int, str, str], ...]:
    return tuple((e.u, e.v, e.pu, e.pv) for e in edges_undirected(s))

def edge_set(s: UBotState) -> Set[Tuple[int, int, str, str]]:
    return set(state_sig(s))

def heuristic(s: UBotState, goal: UBotState) -> int:
    return len(edge_set(s) ^ edge_set(goal))

def _apply_action_pure(s: UBotState, a: Action) -> UBotState:
    s2 = clone_state(s)
    if a.kind == "DETACH":
        remove_edge(s2, a.u, a.v)
    else:
        add_edge(s2, Edge(u=a.u, v=a.v, pu=a.pu, pv=a.pv))
    return s2

def candidate_actions(s: UBotState, goal: UBotState) -> List[Action]:
    cur = edge_set(s)
    g = edge_set(goal)
    actions: List[Action] = []

    # 1) detach extra edges first (keeps branching small)
    for (u, v, _pu, _pv) in sorted(cur - g):
        actions.append(Action(kind="DETACH", u=u, v=v))

    # 2) attach missing goal edges
    for (u, v, pu_goal, pv_goal) in sorted(g - cur):
        free_u = free_ports(s, u)
        free_v = free_ports(s, v)

        if pu_goal in free_u and pv_goal in free_v:
            actions.append(Action(kind="ATTACH", u=u, v=v, pu=pu_goal, pv=pv_goal))
            continue

        # fallback: try any free port pairs (small if you keep ports small)
        for pu in free_u:
            for pv in free_v:
                actions.append(Action(kind="ATTACH", u=u, v=v, pu=pu, pv=pv))

    return actions

@dataclass
class Node:
    f: int
    g: int
    sig: Tuple
    state: UBotState
    path: List[Action]

def plan_astar(start: UBotState, goal: UBotState, max_expand: int = 50000) -> Optional[List[Action]]:
    start_sig = state_sig(start)
    goal_sig = state_sig(goal)
    if start_sig == goal_sig:
        return []

    pq: List[Tuple[int, int, int, Node]] = []
    counter = count()
    h0 = heuristic(start, goal)
    n0 = Node(f=h0, g=0, sig=start_sig, state=start, path=[])
    heappush(pq, (n0.f, n0.g, next(counter), n0))

    best_g: Dict[Tuple, int] = {start_sig: 0}
    expanded = 0

    while pq and expanded < max_expand:
        _f, _g, _idx, node = heappop(pq)

        if node.sig == goal_sig:
            return node.path

        # Skip stale entries
        if node.g != best_g.get(node.sig, node.g):
            continue

        expanded += 1

        for a in candidate_actions(node.state, goal):
            try:
                validate_action(node.state, a)
            except ValidationError:
                continue

            s2 = _apply_action_pure(node.state, a)
            sig2 = state_sig(s2)
            g2 = node.g + 1

            if g2 >= best_g.get(sig2, 10**9):
                continue

            best_g[sig2] = g2
            f2 = g2 + heuristic(s2, goal)
            heappush(pq, (f2, g2, next(counter), Node(f=f2, g=g2, sig=sig2, state=s2, path=node.path + [a])))

    return None
