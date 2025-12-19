from __future__ import annotations
import json
from typing import Any, Dict, List
from ubot.core.types import Edge, UBotState
from ubot.core.graph import empty_state, add_edge
from ubot.core.graph import edges_undirected, compute_parent_map
from ubot.core.validator import validate_joints
from ubot.core.layout import layout_from_ports

def load_state_json(path: str) -> UBotState:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    modules: List[int] = data["modules"]
    ports: List[str] = data.get("ports", ["N", "S", "E", "W"])
    s = empty_state(modules, ports)
    for ed in data.get("edges", []):
        pu = ed["pu"]
        pv = ed["pv"]
        if pu not in ports or pv not in ports:
            raise ValueError(f"Edge uses unknown port: {pu}/{pv}")
        add_edge(s, Edge(u=ed["u"], v=ed["v"], pu=pu, pv=pv))
    joints = data.get("joints")
    if joints is not None:
        # keys might come in as strings; normalize to ints
        s.q = {int(k): list(v) for k, v in joints.items()}
        validate_joints(s.q, modules)
    else:
        s.q = {m: [0.0, 0.0] for m in modules}

    s.parent = compute_parent_map(s, root=0)

    # Initialize / load ax world poses (pose is 12-float tuple: p + R row-major).
    poses_in = data.get("ax_pose")
    if poses_in is not None:
        s.pose = {int(k): tuple(float(x) for x in v["pose"]) for k, v in poses_in.items()}
    else:
        # Deterministic placement from port-layout; uses 0.1 m grid for initial anchors.
        grid = layout_from_ports(s).grid_xy
        s.pose = {}
        for m in modules:
            gx, gy = grid.get(m, (0, 0))
            s.pose[m] = (gx * 0.1, 0.0, gy * 0.1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        s.pose[0] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        # Solve connected component of 0 to satisfy existing edges
        from ubot.core.kinematics import solve_from_anchor, tf_to_pose_tuple, Transform
        import numpy as np

        solved = solve_from_anchor(s, 0, s.pose[0])
        for mid, (p, R) in solved.items():
            s.pose[mid] = tf_to_pose_tuple(Transform(p=np.array(p), R=np.array(R)))

    # Freeze weld transforms for existing edges based on the current poses.
    from ubot.core.kinematics import ensure_welds_from_pose

    ensure_welds_from_pose(s, s.pose)

    # Enforce geometric consistency of existing connections using our kinematics model.
    from ubot.core.kinematics import propagate_poses_3d, edge_geometry_ok
    poses = propagate_poses_3d(s)
    if not edge_geometry_ok(s, poses):
        raise ValueError("State edges violate geometric connection rule (pos coincide + Z opposed).")
    # Enforce "no floating modules/branches" constraints.
    from ubot.core.validator import validate_connectivity

    validate_connectivity(s)
    return s

def dump_state_json(s: UBotState, path: str) -> None:
    data: Dict[str, Any] = {
        "modules": sorted(list(s.adj.keys())),
        "ports": list(s.ports),
        "edges": [e.__dict__ for e in edges_undirected(s)],
    }
    if s.q is not None:
        data["joints"] = {int(k): list(v) for k, v in s.q.items()}
    if s.parent is not None:
        data["parent"] = {int(k): v for k, v in s.parent.items()}
    if s.pose is not None:
        data["ax_pose"] = {int(k): {"pose": list(v)} for k, v in s.pose.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
