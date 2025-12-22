from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

Port = str
EdgeKey = Tuple[int, Port, int, Port]

@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    pu: Port
    pv: Port
    # twist/roll about the connector Z axis (deg), discretized to {0,90,180,270}
    twist: int = 0

@dataclass
class UBotState:
    # adjacency: node -> port -> (other_node, other_port)
    adj: Dict[int, Dict[Port, Tuple[int, Port]]]
    # allowed ports per module (same list for all modules in this minimal repo)
    ports: List[Port]
    # optional: joint positions per module (deg), {module_id: [j1_deg, j2_deg]}
    q: Optional[Dict[int, List[float]]] = None
    # optional: parent map rooted at module 0 ax (module -> parent or None for root/unknown)
    parent: Optional[Dict[int, Optional[int]]] = None
    # optional: world pose of each module's ax as (px,py,pz,r00,r01,r02,r10,r11,r12,r20,r21,r22)
    pose: Optional[Dict[int, Tuple[float, ...]]] = None
    # optional: twist per directed edge (u,pu)->(v,pv), degrees in {0,90,180,270}.
    # For the reverse direction the same twist applies because Rz(t)*Rx(pi) is its own inverse.
    twist: Optional[Dict[EdgeKey, int]] = None

ActionType = Literal["DETACH", "ATTACH", "SET_JOINTS"]

@dataclass(frozen=True)
class Action:
    kind: ActionType
    u: int
    v: int
    pu: Optional[Port] = None
    pv: Optional[Port] = None
    twist: Optional[int] = None
    j1: Optional[float] = None
    j2: Optional[float] = None

    def __str__(self) -> str:
        if self.kind == "DETACH":
            return f"DETACH({self.u},{self.v})"
        if self.kind == "ATTACH":
            if self.twist is None:
                return f"ATTACH({self.u},{self.pu},{self.v},{self.pv})"
            return f"ATTACH({self.u},{self.pu},{self.v},{self.pv},twist={self.twist})"
        return f"SET_JOINTS({self.u},{self.j1},{self.j2})"
