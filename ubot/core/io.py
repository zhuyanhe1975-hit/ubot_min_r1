from __future__ import annotations
import json
from typing import Any, Dict, List
from ubot.core.types import Edge, UBotState
from ubot.core.graph import empty_state, add_edge
from ubot.core.graph import edges_undirected

def load_state_json(path: str) -> UBotState:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    modules: List[int] = data["modules"]
    ports: List[str] = data.get("ports", ["N", "S", "E", "W"])
    s = empty_state(modules, ports)
    for ed in data.get("edges", []):
        add_edge(s, Edge(u=ed["u"], v=ed["v"], pu=ed["pu"], pv=ed["pv"]))
    return s

def dump_state_json(s: UBotState, path: str) -> None:
    data: Dict[str, Any] = {
        "modules": sorted(list(s.adj.keys())),
        "ports": list(s.ports),
        "edges": [e.__dict__ for e in edges_undirected(s)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
