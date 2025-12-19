from __future__ import annotations
from dataclasses import dataclass
from heapq import heappush, heappop
from itertools import count
from typing import Dict, List, Optional, Set, Tuple

from ubot.core.types import UBotState, Action, Edge
from ubot.core.graph import clone_state, edges_undirected, add_edge, remove_edge, free_ports
from ubot.core.validator import validate_action, ValidationError
from ubot.core.kinematics import (
    solve_from_anchor,
    tf_to_pose_tuple,
    Transform,
    ensure_welds_from_pose,
    compute_weld_from_pose,
    tf_inv,
)

def _joints_or_zero(s: UBotState) -> Dict[int, List[float]]:
    if s.q is None:
        return {}
    return s.q

def _edge_sig(s: UBotState) -> Tuple[Tuple[int, int, str, str], ...]:
    return tuple((e.u, e.v, e.pu, e.pv) for e in edges_undirected(s))


def state_sig(s: UBotState) -> Tuple[Tuple[int, int, str, str], ...]:
    edges_sig = _edge_sig(s)
    joints_sig = tuple((m, float(v[0]), float(v[1])) for m, v in sorted(_joints_or_zero(s).items()))
    pose_sig: Tuple[Tuple, ...] = ()
    if s.pose is not None:
        # round poses to keep hashing stable
        pose_sig = tuple((m,) + tuple(round(float(x), 6) for x in s.pose[m]) for m in sorted(s.pose.keys()))
    return edges_sig + joints_sig + pose_sig

def edge_set(s: UBotState) -> Set[Tuple[int, int, str, str]]:
    return set(_edge_sig(s))

def heuristic(s: UBotState, goal: UBotState) -> int:
    h_edges = len(edge_set(s) ^ edge_set(goal))
    goal_q = _joints_or_zero(goal)
    s_q = _joints_or_zero(s)
    h_j = sum(1 for m, vals in goal_q.items() if s_q.get(m, [0.0, 0.0]) != vals)
    return h_edges + h_j

def _apply_action_pure(s: UBotState, a: Action) -> UBotState:
    s2 = clone_state(s)
    if a.kind == "DETACH":
        if s2.weld is not None:
            for pu, (vv, pv) in list(s2.adj.get(a.u, {}).items()):
                if vv == a.v:
                    s2.weld.pop((a.u, pu, a.v, pv), None)
                    s2.weld.pop((a.v, pv, a.u, pu), None)
        remove_edge(s2, a.u, a.v)
    elif a.kind == "ATTACH":
        if s2.pose is not None:
            ensure_welds_from_pose(s2, s2.pose)
            X = compute_weld_from_pose(s2, s2.pose, a.u, a.pu, a.v, a.pv)
            s2.weld[(a.u, a.pu, a.v, a.pv)] = tf_to_pose_tuple(X)
            s2.weld[(a.v, a.pv, a.u, a.pu)] = tf_to_pose_tuple(tf_inv(X))
        add_edge(s2, Edge(u=a.u, v=a.v, pu=a.pu, pv=a.pv))
    else:  # SET_JOINTS
        if s2.q is None:
            s2.q = {}
        s2.q[a.u] = [float(a.j1), float(a.j2)]
    # update parent map and poses
    from ubot.core.graph import compute_parent_map

    s2.parent = compute_parent_map(s2, root=0)
    if s2.pose is not None:
        # update poses component-wise, anchored by existing anchor poses
        modules = list(s2.adj.keys())
        und = {m: set() for m in modules}
        for u, ports in s2.adj.items():
            for _pu, (v, _pv) in ports.items():
                und[u].add(v)
                und[v].add(u)
        seen = set()
        for seed in modules:
            if seed in seen:
                continue
            stack = [seed]
            comp = []
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                comp.append(u)
                stack.extend(list(und.get(u, ())))
            if not comp:
                continue
            anchor = 0 if 0 in comp else min(comp)
            anchor_pose = s2.pose.get(anchor, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            solved = solve_from_anchor(s2, anchor, anchor_pose)
            for mid, (p, R) in solved.items():
                s2.pose[mid] = tf_to_pose_tuple(Transform(p=p, R=R))
    return s2

def candidate_actions(s: UBotState, goal: UBotState) -> List[Action]:
    cur = edge_set(s)
    g = edge_set(goal)
    actions: List[Action] = []

    # 0) joint corrections
    goal_q = _joints_or_zero(goal)
    s_q = _joints_or_zero(s)
    for m, vals_goal in sorted(goal_q.items()):
        cur_vals = s_q.get(m, [0.0, 0.0])
        if cur_vals != vals_goal:
            actions.append(Action(kind="SET_JOINTS", u=m, v=m, j1=vals_goal[0], j2=vals_goal[1]))

    # 0b) explore alternative joint settings (discrete) only for modules
    # that participate in missing goal attachments (keeps branching manageable).
    missing_edges = sorted(g - cur)
    if missing_edges:
        involved = set()
        for (u, v, _pu, _pv) in missing_edges:
            involved.add(u)
            involved.add(v)
        valid_pairs = [
            (-90.0, -90.0),
            (-90.0, 0.0),
            (0.0, -90.0),
            (0.0, 0.0),
            (0.0, 90.0),
            (90.0, 0.0),
            (90.0, 90.0),
        ]
        for m in sorted(involved):
            cur_vals = s_q.get(m, [0.0, 0.0])
            for (j1, j2) in valid_pairs:
                if [j1, j2] != cur_vals:
                    actions.append(Action(kind="SET_JOINTS", u=m, v=m, j1=j1, j2=j2))

    # 1) detach extra edges first (keeps branching small)
    for (u, v, _pu, _pv) in sorted(cur - g):
        actions.append(Action(kind="DETACH", u=u, v=v))

    # 2) attach missing goal edges
    for (u, v, pu_goal, pv_goal) in missing_edges:
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
