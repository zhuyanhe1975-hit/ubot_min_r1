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
    compute_twist_from_pose,
    tf_mul,
    tf_inv,
    port_tf_in_ax,
    weld_tf,
)

def _joints_or_zero(s: UBotState) -> Dict[int, List[float]]:
    if s.q is None:
        return {}
    return s.q

def _edge_sig(s: UBotState) -> Tuple[Tuple[int, int, str, str, int], ...]:
    return tuple((e.u, e.v, e.pu, e.pv, int(e.twist) % 360) for e in edges_undirected(s))


def state_sig(s: UBotState) -> Tuple[Tuple, ...]:
    edges_sig = _edge_sig(s)
    joints_sig = tuple((m, float(v[0]), float(v[1])) for m, v in sorted(_joints_or_zero(s).items()))
    return edges_sig + joints_sig

def edge_set(s: UBotState) -> Set[Tuple[int, int, str, str, int]]:
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
        remove_edge(s2, a.u, a.v)
    elif a.kind == "ATTACH":
        twist = int(a.twist) % 360 if a.twist is not None else None
        if twist is None and s2.pose is not None:
            twist = compute_twist_from_pose(s2, s2.pose, a.u, a.pu, a.v, a.pv)
        add_edge(s2, Edge(u=a.u, v=a.v, pu=a.pu, pv=a.pv, twist=int(twist or 0)))
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
        twist = 0
        for p_a, (bb, p_b) in s.adj.get(a, {}).items():
            if bb == b:
                pu, pv = p_a, p_b
                if s.twist is not None:
                    twist = int(s.twist.get((a, pu, b, pv), 0)) % 360
                break
        if pu is None or pv is None:
            raise ValueError("Broken adjacency while reconstructing path.")
        if a < b:
            edges.append(Edge(u=a, v=b, pu=pu, pv=pv, twist=twist))
        else:
            edges.append(Edge(u=b, v=a, pu=pv, pv=pu, twist=twist))
    return nodes, edges


def _edge_between(s: UBotState, u: int, v: int) -> Optional[Tuple[str, str, int]]:
    """
    If u and v are directly connected, return (pu_on_u, pv_on_v, twist_deg) for the directed u->v.
    """
    for pu, (vv, pv) in s.adj.get(u, {}).items():
        if vv != v:
            continue
        twist = 0 if s.twist is None else int(s.twist.get((u, pu, v, pv), 0)) % 360
        return (pu, pv, twist)
    return None


def _can_detach_without_isolation(s: UBotState, u: int, v: int) -> bool:
    from ubot.core.graph import clone_state, remove_edge
    from ubot.core.validator import validate_connectivity, ValidationError

    s2 = clone_state(s)
    try:
        remove_edge(s2, u, v)
        validate_connectivity(s2)
        return True
    except Exception:
        return False


def _support_loop_port_switch_seq(
    s: UBotState,
    *,
    leaf: int,
    parent: int,
    goal_pu: str,
    goal_pv: str,
    goal_twist: int,
    max_inner_tries: int,
    max_inner_solutions: int,
    apply_seq_fn,
) -> Optional[List[Action]]:
    """
    Try a 4-module temporary-support template to switch the leaf-parent connection ports.

    Precondition:
      - `leaf` is degree-1 and connected to `parent`.
      - Direct DETACH(leaf,parent) is not allowed (would isolate leaf).

    Template (cycle length 4):
      1) pick support node `s3` at distance 3 from leaf along the unique path: leaf-parent-n2-s3
      2) align and ATTACH(s3, ps, leaf, pl) as temporary support (ps/pl are free ports)
      3) DETACH(parent, leaf) (now allowed)
      4) align and ATTACH(parent, goal_pu, leaf, goal_pv, twist=goal_twist)
      5) DETACH(s3, leaf) to remove temporary support
    """
    # Find the 4-node chain leaf -> parent -> n2 -> s3
    path_to_root, _ = _find_path_nodes_and_edges(s, 0, leaf)  # root..leaf
    if len(path_to_root) < 4:
        return None
    # last nodes are ... s3, n2, parent, leaf
    s3, n2, parent2, leaf2 = path_to_root[-4], path_to_root[-3], path_to_root[-2], path_to_root[-1]
    if leaf2 != leaf or parent2 != parent:
        return None
    support = s3

    # Need a path between support and leaf in current graph (should be support-n2-parent-leaf)
    path_nodes, _ = _find_path_nodes_and_edges(s, support, leaf)
    if len(path_nodes) != 4:
        return None

    # Temporary support edge must use currently-free ports.
    free_leaf = free_ports(s, leaf)
    free_sup = free_ports(s, support)
    if not free_leaf or not free_sup:
        return None

    # Try a few port pairs for temporary support.
    port_pairs = [(ps, pl) for ps in free_sup for pl in free_leaf]
    port_pairs = port_pairs[:8]

    for ps, pl in port_pairs:
        joint_maps = _align_ports_by_path_joints_multi(
            s,
            support,
            ps,
            leaf,
            pl,
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
                    joint_actions.append(Action(kind="SET_JOINTS", u=mid, v=mid, j1=float(tgt[0]), j2=float(tgt[1])))

            # Build base state after temporary support attach
            seq1 = joint_actions + [Action(kind="ATTACH", u=support, v=leaf, pu=ps, pv=pl)]
            s1 = None
            try:
                s1 = apply_seq_fn(s, seq1)
            except Exception:
                s1 = None
            if s1 is None:
                continue

            # Detach parent-leaf (should now be legal)
            seq2 = [Action(kind="DETACH", u=parent, v=leaf)]
            s2 = None
            try:
                s2 = apply_seq_fn(s1, seq2)
            except Exception:
                s2 = None
            if s2 is None:
                continue

            # Now align desired ports between parent and leaf along the remaining path parent-n2-support-leaf
            try:
                path_nodes2, _ = _find_path_nodes_and_edges(s2, parent, leaf)
            except Exception:
                continue
            joint_maps2 = _align_ports_by_path_joints_multi(
                s2,
                parent,
                goal_pu,
                leaf,
                goal_pv,
                path_nodes2,
                max_tries=max_inner_tries,
                max_solutions=max_inner_solutions,
            )
            if not joint_maps2:
                continue
            for jm2 in joint_maps2:
                joint_actions2: List[Action] = []
                cur_q2 = _joints_or_zero(s2)
                for mid in sorted(jm2.keys()):
                    tgt = jm2[mid]
                    if cur_q2.get(mid, [0.0, 0.0]) != tgt:
                        joint_actions2.append(
                            Action(kind="SET_JOINTS", u=mid, v=mid, j1=float(tgt[0]), j2=float(tgt[1]))
                        )
                seq3 = joint_actions2 + [
                    Action(kind="ATTACH", u=parent, v=leaf, pu=goal_pu, pv=goal_pv, twist=int(goal_twist) % 360),
                    Action(kind="DETACH", u=support, v=leaf),
                ]
                s3_final = None
                try:
                    s3_final = apply_seq_fn(s2, seq3)
                except Exception:
                    s3_final = None
                if s3_final is None:
                    continue
                return seq1 + seq2 + seq3

    return None


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


def _tf_hash(t: Transform, nd: int = 6) -> Tuple[float, ...]:
    p = tuple(round(float(x), nd) for x in t.p.tolist())
    R = tuple(round(float(x), nd) for x in t.R.reshape(-1).tolist())
    return p + R


def _tf_identity() -> Transform:
    import numpy as np

    return Transform(p=np.zeros(3), R=np.eye(3))


def _ports_aligned_in_uport_frame(T_uport_to_vport: Transform, *, pos_tol: float = 1e-4, cos_tol: float = 0.95) -> bool:
    import numpy as np

    if float(np.linalg.norm(T_uport_to_vport.p)) > pos_tol:
        return False
    z = T_uport_to_vport.R[:, 2]
    if float(np.dot(z, np.array([0.0, 0.0, 1.0]))) > -cos_tol:
        return False
    return True


def _align_ports_by_path_dp(
    s: UBotState,
    u: int,
    pu: str,
    v: int,
    pv: str,
    path_nodes: List[int],
    *,
    max_states_per_layer: int = 2000,
) -> Optional[Dict[int, List[float]]]:
    """
    Faster inner solver for a path: DP over discrete joints along `path_nodes` to align (u,pu) with (v,pv).

    Computes the relative transform from u_port to v_port in u_port frame by chaining per-edge transforms:
      T_ax_{i+1} = T_ax_i * (T_ax_i->ax_{i+1})(q_i, q_{i+1})
    and tests final T_uport->vport for position~=0 and z-opposed.
    """
    if not path_nodes or path_nodes[0] != u or path_nodes[-1] != v:
        return None

    allowed = _allowed_joint_pairs()
    base_q = dict(_joints_or_zero(s))

    # Per-node candidate joint pairs ordered by closeness to current joints
    per_node_allowed: List[List[Tuple[float, float]]] = []
    for mid in path_nodes:
        cur_j1, cur_j2 = (base_q.get(mid, [0.0, 0.0])[0], base_q.get(mid, [0.0, 0.0])[1])

        def score(pair: Tuple[float, float]) -> Tuple[float, float, float]:
            j1, j2 = pair
            return (abs(j1 - cur_j1) + abs(j2 - cur_j2), abs(j1 - cur_j1), abs(j2 - cur_j2))

        per_node_allowed.append(sorted(allowed, key=score))

    # Build directed edges info along the path: for i -> i+1, get ports pi (on i), pj (on j).
    path_ports: List[Tuple[str, str]] = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        pi = pj = None
        for p_a, (bb, p_b) in s.adj.get(a, {}).items():
            if bb == b:
                pi, pj = p_a, p_b
                break
        if pi is None or pj is None:
            return None
        path_ports.append((pi, pj))

    # Cache per-edge transform for joint pairs: (a,pi,b,pj, qa, qb) -> Transform
    edge_cache: Dict[Tuple, Transform] = {}

    def edge_tf(a: int, pi: str, b: int, pj: str, qa: Tuple[float, float], qb: Tuple[float, float]) -> Transform:
        key = (a, pi, b, pj, qa[0], qa[1], qb[0], qb[1])
        if key in edge_cache:
            return edge_cache[key]
        X = weld_tf(s, a, pi, b, pj)
        Ta = port_tf_in_ax(pi, qa[0], qa[1])
        Tb = port_tf_in_ax(pj, qb[0], qb[1])
        Tij = tf_mul(tf_mul(Ta, X), tf_inv(Tb))  # ax_a -> ax_b
        edge_cache[key] = Tij
        return Tij

    # DP layers:
    # layer[idx][q_idx][tf_hash] = backpointer
    # backpointer: (prev_q_idx, prev_tf_hash, q_pair)
    layers: List[Dict[int, Dict[Tuple[float, ...], Tuple[int, Tuple[float, ...], Tuple[float, float]]]]] = []

    # layer 0: at u, T_ax_u->ax_u = I
    layer0: Dict[int, Dict[Tuple[float, ...], Tuple[int, Tuple[float, ...], Tuple[float, float]]]] = {}
    for q_pair in per_node_allowed[0]:
        q_idx = allowed.index(q_pair)
        t0 = _tf_identity()
        layer0.setdefault(q_idx, {})[_tf_hash(t0)] = (-1, (), q_pair)
    layers.append(layer0)

    # forward propagate
    for step, ((a, b), (pi, pj)) in enumerate(zip(zip(path_nodes[:-1], path_nodes[1:]), path_ports), start=0):
        prev_layer = layers[-1]
        next_layer: Dict[int, Dict[Tuple[float, ...], Tuple[int, Tuple[float, ...], Tuple[float, float]]]] = {}

        # cap breadth
        total_prev = sum(len(d) for d in prev_layer.values())
        if total_prev > max_states_per_layer:
            # keep arbitrary subset (stable by insertion)
            kept = 0
            pruned: Dict[int, Dict[Tuple[float, ...], Tuple[int, Tuple[float, ...], Tuple[float, float]]]] = {}
            for q_idx, d in prev_layer.items():
                for th, bp in d.items():
                    pruned.setdefault(q_idx, {})[th] = bp
                    kept += 1
                    if kept >= max_states_per_layer:
                        break
                if kept >= max_states_per_layer:
                    break
            prev_layer = pruned

        for q_idx, tf_map in prev_layer.items():
            qa = allowed[q_idx]
            for tfh_prev, (bp_q, bp_th, bp_pair) in tf_map.items():
                # reconstruct Transform for this hash? store actual Transform is expensive;
                # Instead, keep alongside hash by reusing computed chain using backpointers is complex.
                # Practical compromise: decode back from hash into Transform.
                import numpy as np

                p = np.array(tfh_prev[0:3], dtype=float)
                R = np.array(tfh_prev[3:12], dtype=float).reshape(3, 3)
                T_u_to_a = Transform(p=p, R=R)

                for qb in per_node_allowed[step + 1]:
                    qj = allowed.index(qb)
                    Tij = edge_tf(a, pi, b, pj, qa, qb)
                    T_u_to_b = tf_mul(T_u_to_a, Tij)
                    th = _tf_hash(T_u_to_b)
                    # store first found (ordered by closeness to current, tends to be low-motion)
                    if th not in next_layer.setdefault(qj, {}):
                        next_layer[qj][th] = (q_idx, tfh_prev, qb)

        layers.append(next_layer)

    # terminal check at v
    final_layer = layers[-1]
    if not final_layer:
        return None

    # For each joint state at u and v, we need to compute uport->vport
    # We don't track q_u in DP state explicitly; it's the first layer backpointer pair.
    for qv_idx, tf_map in final_layer.items():
        qv = allowed[qv_idx]
        for tfh, (prev_q_idx, prev_th, qb_pair) in tf_map.items():
            import numpy as np

            p = np.array(tfh[0:3], dtype=float)
            R = np.array(tfh[3:12], dtype=float).reshape(3, 3)
            T_u_to_v = Transform(p=p, R=R)

            # recover q_u from layer0 backpointer (stored in layers[0])
            # layer0 stores bp with q_pair itself; pick matching entry by searching back pointers chain.
            # Backtrack joints for all nodes and then evaluate alignment.
            joints_seq: List[Tuple[float, float]] = [qv]
            cur_q_idx = qv_idx
            cur_th = tfh
            for layer_i in range(len(path_nodes) - 1, 0, -1):
                bp = layers[layer_i][cur_q_idx][cur_th]
                prev_q_idx, prev_th, q_pair = bp
                joints_seq.append(allowed[prev_q_idx] if prev_q_idx >= 0 else q_pair)
                cur_q_idx, cur_th = prev_q_idx, prev_th
            joints_seq.reverse()
            qu = joints_seq[0]

            T_uport = port_tf_in_ax(pu, qu[0], qu[1])
            T_vport = port_tf_in_ax(pv, qv[0], qv[1])
            T_uport_to_vport = tf_mul(tf_mul(tf_inv(T_uport), T_u_to_v), T_vport)
            if not _ports_aligned_in_uport_frame(T_uport_to_vport):
                continue

            return {mid: [float(j1), float(j2)] for mid, (j1, j2) in zip(path_nodes, joints_seq)}

    return None


def compute_tree_edge_swaps(start: UBotState, goal: UBotState) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Compute a deterministic sequence of edge-swaps to transform start tree into goal tree (ignoring port labels).

    Each swap is (add_edge(u,v), remove_edge(a,b)), where remove_edge lies on the created cycle and is not in goal.
    Preconditions (for deterministic behavior):
      - start and goal are connected trees over the same node set.
    """
    def pair_set(s: UBotState) -> Set[Tuple[int, int]]:
        return set((e.u, e.v) for e in edges_undirected(s))

    start_pairs = set(tuple(sorted(p)) for p in pair_set(start))
    goal_pairs = set(tuple(sorted(p)) for p in pair_set(goal))
    if start_pairs == goal_pairs:
        return []
    # Trees: |E| = |V|-1
    if len(start_pairs) != len(start.adj) - 1 or len(goal_pairs) != len(goal.adj) - 1:
        raise ValueError("compute_tree_edge_swaps expects trees (|E|=|V|-1).")

    swaps: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    cur_pairs = set(start_pairs)

    def build_adj(pairs: Set[Tuple[int, int]]) -> Dict[int, List[int]]:
        adj = {m: [] for m in start.adj.keys()}
        for a, b in pairs:
            adj[a].append(b)
            adj[b].append(a)
        return adj

    def find_path(adj: Dict[int, List[int]], src: int, dst: int) -> List[int]:
        from collections import deque

        prev = {src: None}
        q = deque([src])
        while q:
            x = q.popleft()
            if x == dst:
                break
            for y in adj.get(x, []):
                if y in prev:
                    continue
                prev[y] = x
                q.append(y)
        if dst not in prev:
            raise ValueError("Tree path not found (graph disconnected?)")
        nodes: List[int] = []
        cur = dst
        while cur is not None:
            nodes.append(cur)
            cur = prev[cur]
        nodes.reverse()
        return nodes

    while cur_pairs != goal_pairs:
        missing = sorted(list(goal_pairs - cur_pairs))
        extra = set(cur_pairs - goal_pairs)
        if not missing:
            break

        # choose missing edge that yields the shortest cycle (shortest path) to reduce inner complexity
        best = None
        cur_adj = build_adj(cur_pairs)
        for (u, v) in missing:
            path = find_path(cur_adj, u, v)
            plen = len(path)
            cand = (plen, u, v, path)
            if best is None or cand < best:
                best = cand
        assert best is not None
        _plen, u, v, path = best

        # cycle edges are along path (u..v)
        cycle_edges = set(tuple(sorted((a, b))) for a, b in zip(path[:-1], path[1:]))
        removable = sorted(list(cycle_edges & extra))
        if not removable:
            raise ValueError("No removable edge on cycle; goal might be inconsistent for tree transform.")
        # deterministic: remove the first removable edge
        a, b = removable[0]

        swaps.append(((u, v), (a, b)))
        cur_pairs.add(tuple(sorted((u, v))))
        cur_pairs.remove(tuple(sorted((a, b))))

    return swaps


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

    goal_edges = edge_set(goal) if not ignore_ports else set((u, v, "", "", 0) for (u, v) in pair_set(goal))
    goal_q = _joints_or_zero(goal)

    def h(s: UBotState) -> int:
        cur_edges = edge_set(s) if not ignore_ports else set((u, v, "", "", 0) for (u, v) in pair_set(s))
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

    # Deterministic outer topology plan for tree->tree (ignoring ports), then align each swap using DP.
    if ignore_ports:
        from ubot.adapters.sim_adapter import SimAdapter

        try:
            swaps = compute_tree_edge_swaps(start, goal)
        except Exception:
            swaps = []

        adapter = SimAdapter(start)
        plan: List[Action] = []
        for (u, v), (a, b) in swaps:
            st = adapter.get_state()
            # Ports free list (limit tries)
            free_u = free_ports(st, u)
            free_v = free_ports(st, v)
            port_pairs = [(pu, pv) for pu in free_u for pv in free_v]
            if max_port_pairs > 0:
                port_pairs = port_pairs[:max_port_pairs]

            # Path on current tree between u and v
            path_nodes, _path_edges = _find_path_nodes_and_edges(st, u, v)

            best_seq: Optional[List[Action]] = None
            best_cost = 10**9
            for pu, pv in port_pairs:
                jm = _align_ports_by_path_dp(st, u, pu, v, pv, path_nodes)
                if jm is None:
                    continue
                cur_q = _joints_or_zero(st)
                joint_actions: List[Action] = []
                for mid in path_nodes:
                    tgt = jm.get(mid, cur_q.get(mid, [0.0, 0.0]))
                    if cur_q.get(mid, [0.0, 0.0]) != tgt:
                        joint_actions.append(
                            Action(kind="SET_JOINTS", u=mid, v=mid, j1=float(tgt[0]), j2=float(tgt[1]))
                        )
                seq = joint_actions + [Action(kind="ATTACH", u=u, v=v, pu=pu, pv=pv)] + [Action(kind="DETACH", u=a, v=b)]
                s2 = apply_seq(st, seq)
                if s2 is None:
                    continue
                cost = len(seq)
                if cost < best_cost:
                    best_cost = cost
                    best_seq = seq

            if best_seq is None:
                return None
            for act in best_seq:
                adapter.apply(act)
                plan.append(act)

        # Fix joints to goal (still must keep existing geometry valid).
        tail = fix_joints_seq(adapter.get_state())
        if tail is None:
            return None
        for act in tail:
            adapter.apply(act)
            plan.append(act)

        # Final check (ignore ports)
        if set(tuple(sorted(p)) for p in pair_set(adapter.get_state())) != set(tuple(sorted(p)) for p in pair_set(goal)):
            return None
        return plan

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

        cur_edges_now = edge_set(s) if not ignore_ports else set((u, v, "", "", 0) for (u, v) in pair_set(s))
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
        for (u, v, pu_goal, pv_goal, twist_goal) in missing:
            # Special case: u-v already directly connected but with different ports (a "port switch").
            # If detaching would isolate a leaf, try a 4-module temporary-support loop template.
            cur_uv = _edge_between(s, u, v)
            cur_vu = _edge_between(s, v, u)
            if cur_uv is not None or cur_vu is not None:
                # Determine current edge ports and which endpoint is a leaf
                deg_u = len(s.adj.get(u, {}))
                deg_v = len(s.adj.get(v, {}))
                if (deg_u == 1) ^ (deg_v == 1):
                    leaf = u if deg_u == 1 else v
                    parent = v if leaf == u else u
                    # only if detaching isolates (i.e., cannot detach without violating connectivity)
                    if not _can_detach_without_isolation(s, leaf, parent):
                        # goal ports are expressed on u and v; convert to parent/leaf directed form
                        if parent == u and leaf == v:
                            goal_pu2, goal_pv2, goal_tw2 = pu_goal, pv_goal, twist_goal
                        elif parent == v and leaf == u:
                            goal_pu2, goal_pv2, goal_tw2 = pv_goal, pu_goal, twist_goal
                        else:
                            goal_pu2, goal_pv2, goal_tw2 = pu_goal, pv_goal, twist_goal
                        seq = _support_loop_port_switch_seq(
                            s,
                            leaf=leaf,
                            parent=parent,
                            goal_pu=goal_pu2,
                            goal_pv=goal_pv2,
                            goal_twist=int(goal_tw2) % 360,
                            max_inner_tries=max_inner_tries,
                            max_inner_solutions=max_inner_solutions,
                            apply_seq_fn=apply_seq,
                        )
                        if seq is not None:
                            s2 = apply_seq(s, seq)
                            if s2 is not None:
                                sig2 = state_sig(s2)
                                g2 = g_cost + len(seq)
                                if g2 < best_g.get(sig2, 10**9):
                                    best_g[sig2] = g2
                                    heappush(pq, (g2 + h(s2), g2, next(counter), (s2, path + seq)))

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
                removable = [e for e in path_edges if (e.u, e.v, e.pu, e.pv, int(e.twist) % 360) not in goal_edges]
            else:
                goal_pairs = set((uu, vv) for (uu, vv, _a, _b, _t) in goal_edges)
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
                            + [
                                Action(
                                    kind="ATTACH",
                                    u=u,
                                    v=v,
                                    pu=pu,
                                    pv=pv,
                                    twist=None if ignore_ports else int(twist_goal) % 360,
                                )
                            ]
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
        for (u, v, _pu, _pv, _tw) in missing_edges:
            involved.add(u)
            involved.add(v)
        valid_pairs = _allowed_joint_pairs()
        for m in sorted(involved):
            cur_vals = s_q.get(m, [0.0, 0.0])
            for (j1, j2) in valid_pairs:
                if [j1, j2] != cur_vals:
                    actions.append(Action(kind="SET_JOINTS", u=m, v=m, j1=j1, j2=j2))

    # 1) detach extra edges first (keeps branching small)
    for (u, v, _pu, _pv, _tw) in sorted(cur - g):
        actions.append(Action(kind="DETACH", u=u, v=v))

    # 2) attach missing goal edges
    for (u, v, pu_goal, pv_goal, twist_goal) in missing_edges:
        free_u = free_ports(s, u)
        free_v = free_ports(s, v)

        if pu_goal in free_u and pv_goal in free_v:
            actions.append(Action(kind="ATTACH", u=u, v=v, pu=pu_goal, pv=pv_goal, twist=int(twist_goal) % 360))
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
