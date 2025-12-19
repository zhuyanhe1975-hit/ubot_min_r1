from __future__ import annotations
from ubot.adapters.base import UBotAdapter
from ubot.core.types import Action, UBotState, Edge
from ubot.core.graph import clone_state, add_edge, remove_edge, compute_parent_map
from ubot.core.validator import validate_action
from ubot.core.kinematics import (
    solve_from_anchor,
    tf_to_pose_tuple,
    pose_tuple_to_tf,
    Transform,
    ensure_welds_from_pose,
    compute_weld_from_pose,
    tf_inv,
)

class SimAdapter(UBotAdapter):
    """Pure-topology simulator: applies actions to the graph with validation."""

    def __init__(self, initial: UBotState):
        self._state = clone_state(initial)
        # ensure parent map initialized
        self._state.parent = compute_parent_map(self._state, root=0)
        if self._state.pose is None:
            # if caller didn't provide poses, initialize module0 at origin identity and others at origin
            self._state.pose = {m: (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) for m in self._state.adj.keys()}
            self._state.pose[0] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        ensure_welds_from_pose(self._state, self._state.pose)
        self._update_poses()

    def get_state(self) -> UBotState:
        return clone_state(self._state)

    def apply(self, action: Action) -> None:
        validate_action(self._state, action)
        if action.kind == "DETACH":
            # Remove weld entries for the edge before removing it from adjacency.
            if self._state.weld is not None:
                for pu, (vv, pv) in list(self._state.adj.get(action.u, {}).items()):
                    if vv == action.v:
                        self._state.weld.pop((action.u, pu, action.v, pv), None)
                        self._state.weld.pop((action.v, pv, action.u, pu), None)
            remove_edge(self._state, action.u, action.v)
        elif action.kind == "ATTACH":
            add_edge(self._state, Edge(u=action.u, v=action.v, pu=action.pu, pv=action.pv))
            # Freeze weld at the current aligned pose (no teleport).
            if self._state.pose is not None:
                ensure_welds_from_pose(self._state, self._state.pose)
                X = compute_weld_from_pose(self._state, self._state.pose, action.u, action.pu, action.v, action.pv)
                self._state.weld[(action.u, action.pu, action.v, action.pv)] = tf_to_pose_tuple(X)
                self._state.weld[(action.v, action.pv, action.u, action.pu)] = tf_to_pose_tuple(tf_inv(X))
        else:  # SET_JOINTS
            if self._state.q is None:
                self._state.q = {}
            self._state.q[action.u] = [float(action.j1), float(action.j2)]
        # Maintain parent map rooted at module 0 ax
        self._state.parent = compute_parent_map(self._state, root=0)
        # Update ax world poses using kinematic constraints, without teleport.
        self._update_poses()

    def _update_poses(self) -> None:
        """
        Update state.pose in-place.
        - Component containing module0 is anchored at its existing pose (typically origin).
        - Other components are anchored at their existing anchor module pose to avoid jumps.
        """
        if self._state.pose is None:
            return
        modules = list(self._state.adj.keys())

        # build undirected adjacency
        und = {m: set() for m in modules}
        for u, ports in self._state.adj.items():
            for _pu, (v, _pv) in ports.items():
                und[u].add(v)
                und[v].add(u)

        seen = set()
        for seed in modules:
            if seed in seen:
                continue
            # collect component
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

            if 0 in comp:
                anchor = 0
            else:
                anchor = min(comp)
            anchor_pose = self._state.pose.get(anchor)
            if anchor_pose is None:
                anchor_pose = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

            solved = solve_from_anchor(self._state, anchor, anchor_pose)
            for mid, (p, R) in solved.items():
                self._state.pose[mid] = tf_to_pose_tuple(Transform(p=p, R=R))
