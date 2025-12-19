from __future__ import annotations
from ubot.adapters.base import UBotAdapter
from ubot.core.types import Action, UBotState, Edge
from ubot.core.graph import clone_state, add_edge, remove_edge
from ubot.core.validator import validate_action

class SimAdapter(UBotAdapter):
    """Pure-topology simulator: applies actions to the graph with validation."""

    def __init__(self, initial: UBotState):
        self._state = clone_state(initial)

    def get_state(self) -> UBotState:
        return clone_state(self._state)

    def apply(self, action: Action) -> None:
        validate_action(self._state, action)
        if action.kind == "DETACH":
            remove_edge(self._state, action.u, action.v)
        else:
            add_edge(self._state, Edge(u=action.u, v=action.v, pu=action.pu, pv=action.pv))
