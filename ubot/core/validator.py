from __future__ import annotations
from typing import Dict, List
from ubot.core.types import UBotState, Action

class ValidationError(Exception):
    pass

def _connected_to_root(s: UBotState, root: int = 0) -> bool:
    if root not in s.adj:
        return False
    seen = set()
    stack = [root]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for _pu, (v, _pv) in s.adj.get(u, {}).items():
            if v not in seen:
                stack.append(v)
    return len(seen) == len(s.adj)


def _no_isolated_modules(s: UBotState) -> bool:
    if len(s.adj) <= 1:
        return True
    for m in s.adj.keys():
        if len(s.adj.get(m, {})) == 0:
            return False
    return True


def validate_connectivity(s: UBotState) -> None:
    """Hard constraint: all modules must remain connected to root=0; no isolated modules."""
    if not _connected_to_root(s, root=0):
        raise ValidationError("Disconnected component detected: all modules must have a path to root=0.")
    if not _no_isolated_modules(s):
        raise ValidationError("Isolated module detected: every module must be connected to at least one other module.")


def validate_action(s: UBotState, a: Action) -> None:
    if a.kind == "DETACH":
        if a.u == a.v:
            raise ValidationError("u and v must differ.")
        ports = s.adj.get(a.u, {})
        ok = any(v == a.v for (v, _pv) in ports.values())
        if not ok:
            raise ValidationError(f"Edge {a.u}-{a.v} not connected.")
        # Hard constraint: detaching must not disconnect from root or isolate a module.
        from ubot.core.graph import clone_state, remove_edge

        s2 = clone_state(s)
        if s2.weld is not None:
            for pu, (vv, pv) in list(s2.adj.get(a.u, {}).items()):
                if vv == a.v:
                    s2.weld.pop((a.u, pu, a.v, pv), None)
                    s2.weld.pop((a.v, pv, a.u, pu), None)
        remove_edge(s2, a.u, a.v)
        validate_connectivity(s2)
    elif a.kind == "ATTACH":
        if a.u == a.v:
            raise ValidationError("u and v must differ.")
        if a.pu is None or a.pv is None:
            raise ValidationError("ATTACH requires pu and pv.")
        if a.pu not in s.ports or a.pv not in s.ports:
            raise ValidationError("Unknown port name.")
        if a.pu in s.adj.get(a.u, {}):
            raise ValidationError(f"Port occupied: {a.u}.{a.pu}")
        if a.pv in s.adj.get(a.v, {}):
            raise ValidationError(f"Port occupied: {a.v}.{a.pv}")
        # Geometric constraint: ports must already be aligned in world (no teleport).
        if s.pose is None:
            raise ValidationError("Missing state pose; cannot validate geometric ATTACH.")
        from ubot.core.kinematics import ports_aligned

        if not ports_aligned(s, s.pose, a.u, a.pu, a.v, a.pv):
            raise ValidationError("Ports not aligned in world for ATTACH.")
    elif a.kind == "SET_JOINTS":
        if a.j1 is None or a.j2 is None:
            raise ValidationError("SET_JOINTS requires j1/j2.")
        validate_joints({a.u: [a.j1, a.j2]}, modules=[a.u])
        # Kinematic feasibility: after changing joints, existing edges must remain geometrically valid.
        if s.pose is None:
            raise ValidationError("Missing state pose; cannot validate kinematic SET_JOINTS.")
        from ubot.core.graph import clone_state, compute_parent_map
        from ubot.core.kinematics import solve_from_anchor, tf_to_pose_tuple, Transform, edge_geometry_ok, propagate_poses_3d
        import numpy as np

        s2 = clone_state(s)
        s2.q = dict(s2.q or {})
        s2.q[a.u] = [float(a.j1), float(a.j2)]
        s2.parent = compute_parent_map(s2, root=0)
        solved = solve_from_anchor(s2, 0, s.pose[0])
        for mid, (p, R) in solved.items():
            s2.pose[mid] = tf_to_pose_tuple(Transform(p=np.array(p), R=np.array(R)))
        poses = propagate_poses_3d(s2)
        if not edge_geometry_ok(s2, poses):
            raise ValidationError("SET_JOINTS would break existing geometric connections.")
    else:
        raise ValidationError("Unknown action kind.")


def validate_joints(joints: dict[int, list[float]], modules: list[int]) -> None:
    """
    Conventions:
      - Each module has two hinge joints (j1, j2), degrees in {-90, -45, 0, 45, 90}
      - If both are non-zero,要求同号（之前的反号约束取消）
      - If either is zero, the other can be any allowed value
    """
    allowed = {-90.0, -45.0, 0.0, 45.0, 90.0}
    for m in modules:
        if m not in joints:
            raise ValidationError(f"Missing joints for module {m}")
        vals = joints[m]
        if len(vals) != 2:
            raise ValidationError(f"Module {m} joints must have length 2")
        j1, j2 = float(vals[0]), float(vals[1])
        if j1 not in allowed or j2 not in allowed:
            raise ValidationError(f"Module {m} joints must be in {-90,-45,0,45,90}, got ({j1},{j2})")
        if j1 != 0.0 and j2 != 0.0 and j1 != j2:
            raise ValidationError(f"Module {m} joints must have same sign when both non-zero, got ({j1},{j2})")
