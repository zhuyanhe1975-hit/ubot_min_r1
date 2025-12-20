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
    weld_sig: Tuple[Tuple, ...] = ()
    if s.weld is not None:
        weld_sig = tuple(
            (u, pu, v, pv) + tuple(round(float(x), 6) for x in tf)
            for (u, pu, v, pv), tf in sorted(s.weld.items())
        )
    return edges_sig + joints_sig + pose_sig + weld_sig

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
    if s2.pose is not None and a.kind == "SET_JOINTS":
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


def _allowed_joint_pairs() -> List[Tuple[float, float]]:
    # Must match validator.validate_joints
    return [
        (-90.0, -90.0),
        (-90.0, 0.0),
        (0.0, -90.0),
        (0.0, 0.0),
        (0.0, 90.0),
        (90.0, 0.0),
        (90.0, 90.0),
    ]


def _find_path_nodes_and_edges(s: UBotState, u: int, v: int) -> Tuple[List[int], List[Edge]]:
    """
    Return (nodes_on_path, edges_on_path) for the unique path in a connected graph.
    For our demos, the intermediate topology is intended to stay a tree (or temporarily a single-cycle),
    but we only call this on the current tree before an edge-swap.
    """
    if u == v:
        return [u], []

    from collections import deque

    prev: Dict[int, Optional[int]] = {u: None}
    q = deque([u])
    while q:
        x = q.popleft()
        if x == v:
            break
        for _px, (y, _py) in s.adj.get(x, {}).items():
            if y in prev:
                continue
            prev[y] = x
            q.append(y)
    if v not in prev:
        raise ValueError(f"No path between {u} and {v}")

    nodes: List[int] = []
    cur = v
    while cur is not None:
        nodes.append(cur)
        cur = prev[cur]
    nodes.reverse()

    edges: List[Edge] = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        pu = pv = None
        for p_a, (bb, p_b) in s.adj.get(a, {}).items():
            if bb == b:
                pu, pv = p_a, p_b
                break
        if pu is None or pv is None:
            raise ValueError("Broken adjacency while reconstructing path.")
        if a < b:
            edges.append(Edge(u=a, v=b, pu=pu, pv=pv))
        else:
            edges.append(Edge(u=b, v=a, pu=pv, pv=pu))
    return nodes, edges


def _align_ports_by_path_joints(
    s: UBotState,
    u: int,
    pu: str,
    v: int,
    pv: str,
    path_nodes: List[int],
    *,
    max_tries: int = 300000,
) -> Optional[Dict[int, List[float]]]:
    """
    Brute force discrete joints on `path_nodes` only, searching for a configuration where
    ports (u,pu) and (v,pv) are aligned in world while preserving all existing connections.
    """
    if s.pose is None:
        return None
    from itertools import product
    from ubot.core.kinematics import ports_aligned, propagate_poses_3d, edge_geometry_ok, solve_from_anchor
    import numpy as np

    allowed = _allowed_joint_pairs()
    base_q = dict(_joints_or_zero(s))
    path_nodes = list(dict.fromkeys(path_nodes))  # stable unique

    tries = 0
    for combo in product(allowed, repeat=len(path_nodes)):
        tries += 1
        if tries > max_tries:
            return None

        s2 = clone_state(s)
        s2.q = dict(base_q)
        for mid, (j1, j2) in zip(path_nodes, combo):
            s2.q[mid] = [float(j1), float(j2)]

        # Solve poses anchored at ax0 pose
        solved = solve_from_anchor(s2, 0, s.pose[0])
        for mid, (p, R) in solved.items():
            s2.pose[mid] = tf_to_pose_tuple(Transform(p=np.array(p), R=np.array(R)))

        poses = propagate_poses_3d(s2)
        if not edge_geometry_ok(s2, poses):
            continue
        if not ports_aligned(s2, s2.pose, u, pu, v, pv):
            continue
        return {mid: list(s2.q[mid]) for mid in path_nodes}

    return None


def _align_ports_by_path_joints_multi(
    s: UBotState,
    u: int,
    pu: str,
    v: int,
    pv: str,
    path_nodes: List[int],
    *,
    max_tries: int = 300000,
    max_solutions: int = 8,
) -> List[Dict[int, List[float]]]:
    """
    Return multiple joint assignments (up to `max_solutions`) that align the given ports,
    enumerating discrete joint combinations on `path_nodes`.
    """
    if max_solutions <= 0:
        return []
    if s.pose is None:
        return []

    from itertools import product
    from ubot.core.kinematics import ports_aligned, propagate_poses_3d, edge_geometry_ok, solve_from_anchor
    import numpy as np

    allowed = _allowed_joint_pairs()
    base_q = dict(_joints_or_zero(s))
    path_nodes = list(dict.fromkeys(path_nodes))  # stable unique

    out: List[Dict[int, List[float]]] = []
    tries = 0

    # Order per-node candidates close to current joints to find low-motion solutions early.
    per_node_allowed: List[List[Tuple[float, float]]] = []
    for mid in path_nodes:
        cur_j1, cur_j2 = (base_q.get(mid, [0.0, 0.0])[0], base_q.get(mid, [0.0, 0.0])[1])

        def score(pair: Tuple[float, float]) -> Tuple[float, float, float]:
            j1, j2 = pair
            return (abs(j1 - cur_j1) + abs(j2 - cur_j2), abs(j1 - cur_j1), abs(j2 - cur_j2))

        per_node_allowed.append(sorted(allowed, key=score))

    for combo in product(*per_node_allowed):
        tries += 1
        if tries > max_tries:
            break

        s2 = clone_state(s)
        s2.q = dict(base_q)
        for mid, (j1, j2) in zip(path_nodes, combo):
            s2.q[mid] = [float(j1), float(j2)]

        solved = solve_from_anchor(s2, 0, s.pose[0])
        for mid, (p, R) in solved.items():
            s2.pose[mid] = tf_to_pose_tuple(Transform(p=np.array(p), R=np.array(R)))

        poses = propagate_poses_3d(s2)
        if not edge_geometry_ok(s2, poses):
            continue
        if not ports_aligned(s2, s2.pose, u, pu, v, pv):
            continue

        out.append({mid: list(s2.q[mid]) for mid in path_nodes})
        if len(out) >= max_solutions:
            break

    return out


def plan_hierarchical_edge_swap(
    start: UBotState,
    goal: UBotState,
    *,
    max_swaps: int = 50,
    max_inner_tries: int = 300000,
    max_inner_solutions: int = 3,
    ignore_ports: bool = False,
    max_port_pairs: int = 4,
) -> Optional[List[Action]]:
    """
    Hierarchical planner (outer: topology via edge-swap; inner: joint search on the u-v path for each ATTACH).

    Designed for tree->tree tasks under strong constraints:
      - no floating modules/branches
      - ATTACH requires pre-aligned ports
      - SET_JOINTS must keep all existing edges geometrically valid
    """
    def pair_set(s: UBotState) -> Set[Tuple[int, int]]:
        return set((e.u, e.v) for e in edges_undirected(s))

    goal_edges = edge_set(goal) if not ignore_ports else set((u, v, "", "") for (u, v) in pair_set(goal))
    goal_q = _joints_or_zero(goal)

    def h(s: UBotState) -> int:
        cur_edges = edge_set(s) if not ignore_ports else set((u, v, "", "") for (u, v) in pair_set(s))
        cur_q = _joints_or_zero(s)
        return len(cur_edges ^ goal_edges) + sum(1 for m, v in goal_q.items() if cur_q.get(m, [0.0, 0.0]) != v)

    def apply_seq(s: UBotState, seq: List[Action]) -> Optional[UBotState]:
        s2 = clone_state(s)
        for a in seq:
            try:
                validate_action(s2, a)
            except ValidationError:
                return None
            s2 = _apply_action_pure(s2, a)
        return s2

    def fix_joints_seq(s: UBotState) -> Optional[List[Action]]:
        cur_q = _joints_or_zero(s)
        seq: List[Action] = []
        for mid in sorted(goal_q.keys()):
            if cur_q.get(mid, [0.0, 0.0]) != goal_q[mid]:
                seq.append(
                    Action(
                        kind="SET_JOINTS",
                        u=mid,
                        v=mid,
                        j1=float(goal_q[mid][0]),
                        j2=float(goal_q[mid][1]),
                    )
                )
        s2 = apply_seq(s, seq)
        return seq if s2 is not None else None

    # Fast path for tree->tree topology problems: bounded DFS over edge-swaps.
    if ignore_ports:
        goal_pairs = set((u, v) for (u, v, _a, _b) in goal_edges)

        def sig_pairs_joints(s: UBotState) -> Tuple:
            pairs = tuple(sorted(pair_set(s)))
            joints = tuple((m, float(v[0]), float(v[1])) for m, v in sorted(_joints_or_zero(s).items()))
            return pairs + joints

        def dfs(s: UBotState, depth: int, seen: Set[Tuple]) -> Optional[List[Action]]:
            cur_pairs = pair_set(s)
            if cur_pairs == goal_pairs:
                tail = fix_joints_seq(s)
                return tail if tail is not None else None
            if depth <= 0:
                return None

            key = sig_pairs_joints(s)
            if key in seen:
                return None
            seen.add(key)

            missing_pairs = sorted(list(goal_pairs - cur_pairs))
            extra_pairs = set(cur_pairs - goal_pairs)

            # Prefer shorter paths first.
            scored = []
            for (u, v) in missing_pairs:
                pn, pe = _find_path_nodes_and_edges(s, u, v)
                scored.append((len(pn), u, v, pn, pe))
            scored.sort()

            for _plen, u, v, path_nodes, path_edges in scored:
                free_u = free_ports(s, u)
                free_v = free_ports(s, v)
                port_pairs = [(pu, pv) for pu in free_u for pv in free_v]
                if max_port_pairs > 0:
                    port_pairs = port_pairs[:max_port_pairs]

                removable = [e for e in path_edges if (e.u, e.v) in extra_pairs]
                if not removable:
                    continue

                for (pu, pv) in port_pairs:
                    if pu in s.adj.get(u, {}):
                        continue
                    if pv in s.adj.get(v, {}):
                        continue

                    joint_maps = _align_ports_by_path_joints_multi(
                        s,
                        u,
                        pu,
                        v,
                        pv,
                        path_nodes,
                        max_tries=max_inner_tries,
                        max_solutions=max_inner_solutions,
                    )
                    if not joint_maps:
                        continue

                    for joints_map in joint_maps:
                        joint_actions: List[Action] = []
                        cur_q = _joints_or_zero(s)
                        for mid in sorted(joints_map.keys()):
                            tgt = joints_map[mid]
                            if cur_q.get(mid, [0.0, 0.0]) != tgt:
                                joint_actions.append(
                                    Action(kind="SET_JOINTS", u=mid, v=mid, j1=float(tgt[0]), j2=float(tgt[1]))
                                )

                        for e_remove in removable:
                            seq = (
                                joint_actions
                                + [Action(kind="ATTACH", u=u, v=v, pu=pu, pv=pv)]
                                + [Action(kind="DETACH", u=e_remove.u, v=e_remove.v)]
                            )
                            s2 = apply_seq(s, seq)
                            if s2 is None:
                                continue
                            tail = dfs(s2, depth - 1, seen)
                            if tail is not None:
                                return seq + tail

            return None

        # Depth heuristic: each swap can fix at most one missing pair.
        depth0 = min(max_swaps, len(goal_pairs - pair_set(start)) + 2)
        return dfs(start, depth0, set())

    start_sig = state_sig(start)
    pq: List[Tuple[int, int, int, Tuple[UBotState, List[Action]]]] = []
    counter = count()
    heappush(pq, (h(start), 0, next(counter), (start, [])))
    best_g: Dict[Tuple, int] = {start_sig: 0}
    expanded = 0

    while pq and expanded < max_swaps:
        _f, g_cost, _idx, (s, path) = heappop(pq)
        sig = state_sig(s)
        if g_cost != best_g.get(sig, g_cost):
            continue
        expanded += 1

        cur_edges_now = edge_set(s) if not ignore_ports else set((u, v, "", "") for (u, v) in pair_set(s))
        if cur_edges_now == goal_edges and _joints_or_zero(s) == goal_q:
            return path

        cur_edges = cur_edges_now
        missing = sorted(goal_edges - cur_edges)

        # If topology matches, only fix joints.
        if not missing:
            for mid, vals in sorted(goal_q.items()):
                cur = _joints_or_zero(s).get(mid, [0.0, 0.0])
                if cur == vals:
                    continue
                seq = [Action(kind="SET_JOINTS", u=mid, v=mid, j1=float(vals[0]), j2=float(vals[1]))]
                s2 = apply_seq(s, seq)
                if s2 is None:
                    continue
                sig2 = state_sig(s2)
                g2 = g_cost + len(seq)
                if g2 >= best_g.get(sig2, 10**9):
                    continue
                best_g[sig2] = g2
                heappush(pq, (g2 + h(s2), g2, next(counter), (s2, path + seq)))
            continue

        # Topology step: try edge-swap moves toward missing goal edges.
        for (u, v, pu_goal, pv_goal) in missing:
            # When ignoring ports, treat pu/pv from goal as a hint only.
            if not ignore_ports:
                port_pairs = [(pu_goal, pv_goal)]
            else:
                free_u = free_ports(s, u)
                free_v = free_ports(s, v)
                port_pairs = [(pu, pv) for pu in free_u for pv in free_v]
                # Prefer goal hint first if it happens to be free.
                if pu_goal in free_ports(s, u) and pv_goal in free_ports(s, v):
                    port_pairs = [(pu_goal, pv_goal)] + [pp for pp in port_pairs if pp != (pu_goal, pv_goal)]
                if max_port_pairs > 0:
                    port_pairs = port_pairs[:max_port_pairs]

            path_nodes, path_edges = _find_path_nodes_and_edges(s, u, v)

            # Removable edges must lie on the created cycle and should not be required by goal.
            if not ignore_ports:
                removable = [e for e in path_edges if (e.u, e.v, e.pu, e.pv) not in goal_edges]
            else:
                goal_pairs = set((uu, vv) for (uu, vv, _a, _b) in goal_edges)
                removable = [e for e in path_edges if (e.u, e.v) not in goal_pairs]
            if not removable:
                continue

            for (pu, pv) in port_pairs:
                if pu in s.adj.get(u, {}):
                    continue
                if pv in s.adj.get(v, {}):
                    continue

                joint_maps = _align_ports_by_path_joints_multi(
                    s,
                    u,
                    pu,
                    v,
                    pv,
                    path_nodes,
                    max_tries=max_inner_tries,
                    max_solutions=max_inner_solutions,
                )
                if not joint_maps:
                    continue

                for joints_map in joint_maps:
                    joint_actions: List[Action] = []
                    cur_q = _joints_or_zero(s)
                    for mid in sorted(joints_map.keys()):
                        tgt = joints_map[mid]
                        if cur_q.get(mid, [0.0, 0.0]) != tgt:
                            joint_actions.append(
                                Action(kind="SET_JOINTS", u=mid, v=mid, j1=float(tgt[0]), j2=float(tgt[1]))
                            )

                    for e_remove in removable:
                        seq = (
                            joint_actions
                            + [Action(kind="ATTACH", u=u, v=v, pu=pu, pv=pv)]
                            + [Action(kind="DETACH", u=e_remove.u, v=e_remove.v)]
                        )
                        s2 = apply_seq(s, seq)
                        if s2 is None:
                            continue
                        sig2 = state_sig(s2)
                        g2 = g_cost + len(seq)
                        if g2 >= best_g.get(sig2, 10**9):
                            continue
                        best_g[sig2] = g2
                        heappush(pq, (g2 + h(s2), g2, next(counter), (s2, path + seq)))

    return None

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
    goal_edges = edge_set(goal)
    goal_q = _joints_or_zero(goal)
    if edge_set(start) == goal_edges and _joints_or_zero(start) == goal_q:
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

        if edge_set(node.state) == goal_edges and _joints_or_zero(node.state) == goal_q:
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
