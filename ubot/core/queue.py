from __future__ import annotations
import json
from dataclasses import asdict
from typing import List, Dict, Any, Optional
from ubot.core.types import Action

def save_log(path: str, plan: List[Action], executed: List[Action], meta: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "meta": meta or {},
        "plan": [asdict(a) for a in plan],
        "executed": [asdict(a) for a in executed],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
