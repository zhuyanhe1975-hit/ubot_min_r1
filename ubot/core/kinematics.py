from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ubot.core.types import UBotState
from ubot.core.graph import compute_parent_map, edges_undirected


def rot_x(a_deg: float) -> np.ndarray:
    a = math.radians(a_deg)
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])


def rot_z(a_deg: float) -> np.ndarray:
    a = math.radians(a_deg)
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])


def quat_wxyz_to_mat(q: Tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


@dataclass(frozen=True)
class Transform:
    p: np.ndarray  # (3,)
    R: np.ndarray  # (3,3)


def tf_mul(a: Transform, b: Transform) -> Transform:
    return Transform(p=a.p + a.R @ b.p, R=a.R @ b.R)


def tf_inv(t: Transform) -> Transform:
    Rt = t.R.T
    return Transform(p=-(Rt @ t.p), R=Rt)


def port_site_local(port: str) -> Transform:
    """
    Local site transform in the corresponding half-body frame at zero joint angle.
    Values are copied from `models/ubot/ubot_body.xml`.
    """
    if port == "E":  # ma_connector_right
        return Transform(p=np.array([0.05, 0.0, 0.0]), R=quat_wxyz_to_mat((0.70710678, 0.0, 0.70710678, 0.0)))
    if port == "S":  # ma_connector_bottom
        return Transform(p=np.array([0.0, 0.0, -0.05]), R=quat_wxyz_to_mat((0.0, 0.0, 1.0, 0.0)))
    if port == "W":  # mb_connector_left
        return Transform(p=np.array([-0.05, 0.0, 0.0]), R=quat_wxyz_to_mat((0.70710678, 0.0, -0.70710678, 0.0)))
    if port == "N":  # mb_connector_top
        return Transform(p=np.array([0.0, 0.0, 0.05]), R=np.eye(3))
    raise KeyError(f"Unknown port {port}")


def port_tf_in_ax(port: str, j1: float, j2: float) -> Transform:
    """
    Transform from module ax frame to the given connector site, under current joint angles.
    - ma ports (E,S): hinge about +X by j1
    - mb ports (W,N): hinge about +Z by j2
    """
    if port in ("E", "S"):
        Rj = rot_x(j1)
    else:
        Rj = rot_z(j2)
    tj = Transform(p=np.zeros(3), R=Rj)
    return tf_mul(tj, port_site_local(port))


_FLIP_Y_PI = Transform(p=np.zeros(3), R=np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]))


def pose_tuple_to_tf(pose: Tuple[float, ...]) -> Transform:
    if len(pose) != 12:
        raise ValueError(f"pose must have 12 floats, got {len(pose)}")
    p = np.array(pose[0:3], dtype=float)
    R = np.array(pose[3:12], dtype=float).reshape(3, 3)
    return Transform(p=p, R=R)


def tf_to_pose_tuple(t: Transform) -> Tuple[float, ...]:
    p = [float(x) for x in t.p.tolist()]
    R = [float(x) for x in t.R.reshape(-1).tolist()]
    return tuple(p + R)

def _weld_key(u: int, pu: str, v: int, pv: str) -> tuple[int, str, int, str]:
    return (int(u), str(pu), int(v), str(pv))


def weld_tf(state: UBotState, u: int, pu: str, v: int, pv: str) -> Transform:
    """
    Return the stored weld transform X such that:
      T_world(v_port) == T_world(u_port) * X
    If absent, falls back to the legacy fixed flip (RotY(pi)).
    """
    if state.weld is not None:
        key = _weld_key(u, pu, v, pv)
        if key in state.weld:
            return pose_tuple_to_tf(state.weld[key])
    return _FLIP_Y_PI


def compute_weld_from_pose(state: UBotState, pose_map: Dict[int, Tuple[float, ...]], u: int, pu: str, v: int, pv: str) -> Transform:
    """
    Compute X such that T_world(v_port) = T_world(u_port) * X under the provided pose map.
    """
    Tu = port_world_tf(state, pose_map, u, pu)
    Tv = port_world_tf(state, pose_map, v, pv)
    return tf_mul(tf_inv(Tu), Tv)


def ensure_welds_from_pose(state: UBotState, pose_map: Dict[int, Tuple[float, ...]]) -> None:
    """
    Ensure `state.weld` exists and is populated for all existing edges, using the provided poses.
    This is used to (a) load welds from JSON `ax_pose`, and (b) freeze the attachment twist at ATTACH time.
    """
    if state.weld is None:
        state.weld = {}
    for e in edges_undirected(state):
        key_uv = _weld_key(e.u, e.pu, e.v, e.pv)
        key_vu = _weld_key(e.v, e.pv, e.u, e.pu)
        if key_uv in state.weld and key_vu in state.weld:
            continue
        X = compute_weld_from_pose(state, pose_map, e.u, e.pu, e.v, e.pv)
        state.weld[key_uv] = tf_to_pose_tuple(X)
        state.weld[key_vu] = tf_to_pose_tuple(tf_inv(X))


def solve_from_anchor(
    state: UBotState, anchor: int, anchor_pose: Tuple[float, ...]
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Solve ax world poses for the connected component containing `anchor`, keeping anchor pose fixed.
    """
    parents = compute_parent_map(state, root=anchor)
    poses: Dict[int, Transform] = {anchor: pose_tuple_to_tf(anchor_pose)}

    from collections import deque

    q = deque([anchor])
    while q:
        u = q.popleft()
        Tu = poses[u]
        j1_u, j2_u = (state.q.get(u, [0.0, 0.0]) if state.q else [0.0, 0.0])

        for child, parent in parents.items():
            if parent != u:
                continue
            edge_port = None
            for pu, (v, pv) in state.adj.get(u, {}).items():
                if v == child:
                    edge_port = (pu, pv)
                    break
            if edge_port is None:
                continue

            pu, pv = edge_port
            j1_v, j2_v = (state.q.get(child, [0.0, 0.0]) if state.q else [0.0, 0.0])

            T_u_port = tf_mul(Tu, port_tf_in_ax(pu, j1_u, j2_u))
            T_child_port_in_ax = port_tf_in_ax(pv, j1_v, j2_v)
            X = weld_tf(state, u, pu, child, pv)
            Tv = tf_mul(tf_mul(T_u_port, X), tf_inv(T_child_port_in_ax))
            poses[child] = Tv
            q.append(child)

    return {m: (t.p, t.R) for m, t in poses.items()}


def port_world_tf(state: UBotState, pose_map: Dict[int, Tuple[float, ...]], module: int, port: str) -> Transform:
    T_ax = pose_tuple_to_tf(pose_map[module])
    j1, j2 = (state.q.get(module, [0.0, 0.0]) if state.q else [0.0, 0.0])
    return tf_mul(T_ax, port_tf_in_ax(port, j1, j2))


def ports_aligned(
    state: UBotState,
    pose_map: Dict[int, Tuple[float, ...]],
    u: int,
    pu: str,
    v: int,
    pv: str,
    *,
    pos_tol: float = 1e-4,
    cos_tol: float = 0.95,
) -> bool:
    Tu = port_world_tf(state, pose_map, u, pu)
    Tv = port_world_tf(state, pose_map, v, pv)
    if float(np.linalg.norm(Tu.p - Tv.p)) > pos_tol:
        return False
    zu = Tu.R[:, 2]
    zv = Tv.R[:, 2]
    if float(np.dot(zu, zv)) > -cos_tol:
        return False
    return True


def propagate_poses_3d(state: UBotState, root_z: float = 0.0) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Kinematic solve of module ax world poses using attachment weld constraints.

    - Root ax0 is fixed in world: position (0,root_z,0), rotation = Identity.
    - Internal joints rotate ma/mb, affecting port transforms relative to ax.
    - For an attachment edge u.pu -- v.pv, enforce:
        T_world(v_port) = T_world(u_port) * X_uv
      where X_uv is stored in `state.weld` (frozen at ATTACH time). If missing, uses legacy RotY(pi).
    """
    if state.pose is None or 0 not in state.pose:
        root_pose = tf_to_pose_tuple(Transform(p=np.array([0.0, root_z, 0.0]), R=np.eye(3)))
    else:
        root_pose = state.pose[0]
    return solve_from_anchor(state, 0, root_pose)


def edge_geometry_ok(
    state: UBotState,
    poses: Dict[int, Tuple[np.ndarray, np.ndarray]],
    *,
    pos_tol: float = 1e-4,
    cos_tol: float = 0.95,
) -> bool:
    """
    Check each edge satisfies:
      - connector world positions coincide
      - connector +Z axes oppose (dot < -cos_tol)
    under the solved ax poses and current joints.
    """
    for e in edges_undirected(state):
        p_u, R_u = poses[e.u]
        p_v, R_v = poses[e.v]
        j1_u, j2_u = (state.q.get(e.u, [0.0, 0.0]) if state.q else [0.0, 0.0])
        j1_v, j2_v = (state.q.get(e.v, [0.0, 0.0]) if state.q else [0.0, 0.0])
        Tu = Transform(p=p_u, R=R_u)
        Tv = Transform(p=p_v, R=R_v)
        Tu_port = tf_mul(Tu, port_tf_in_ax(e.pu, j1_u, j2_u))
        Tv_port = tf_mul(Tv, port_tf_in_ax(e.pv, j1_v, j2_v))
        if float(np.linalg.norm(Tu_port.p - Tv_port.p)) > pos_tol:
            return False
        zu = Tu_port.R[:, 2]
        zv = Tv_port.R[:, 2]
        if float(np.dot(zu, zv)) > -cos_tol:
            return False
    return True
