from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

Port = str

@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    pu: Port
    pv: Port

@dataclass
class UBotState:
    # adjacency: node -> port -> (other_node, other_port)
    adj: Dict[int, Dict[Port, Tuple[int, Port]]]
    # allowed ports per module (same list for all modules in this minimal repo)
    ports: List[Port]
    # optional: joint positions, base pose...
    q: Optional[List[float]] = None

ActionType = Literal["DETACH", "ATTACH"]

@dataclass(frozen=True)
class Action:
    kind: ActionType
    u: int
    v: int
    pu: Optional[Port] = None
    pv: Optional[Port] = None

    def __str__(self) -> str:
        if self.kind == "DETACH":
            return f"DETACH({self.u},{self.v})"
        return f"ATTACH({self.u},{self.pu},{self.v},{self.pv})"
