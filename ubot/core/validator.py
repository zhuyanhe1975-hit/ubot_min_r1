from __future__ import annotations
from ubot.core.types import UBotState, Action

class ValidationError(Exception):
    pass

def validate_action(s: UBotState, a: Action) -> None:
    if a.u == a.v:
        raise ValidationError("u and v must differ.")
    if a.kind == "DETACH":
        ports = s.adj.get(a.u, {})
        ok = any(v == a.v for (v, _pv) in ports.values())
        if not ok:
            raise ValidationError(f"Edge {a.u}-{a.v} not connected.")
    elif a.kind == "ATTACH":
        if a.pu is None or a.pv is None:
            raise ValidationError("ATTACH requires pu and pv.")
        if a.pu not in s.ports or a.pv not in s.ports:
            raise ValidationError("Unknown port name.")
        if a.pu in s.adj.get(a.u, {}):
            raise ValidationError(f"Port occupied: {a.u}.{a.pu}")
        if a.pv in s.adj.get(a.v, {}):
            raise ValidationError(f"Port occupied: {a.v}.{a.pv}")
    else:
        raise ValidationError("Unknown action kind.")
