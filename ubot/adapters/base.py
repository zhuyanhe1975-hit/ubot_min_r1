from __future__ import annotations
from abc import ABC, abstractmethod
from ubot.core.types import Action, UBotState

class UBotAdapter(ABC):
    @abstractmethod
    def get_state(self) -> UBotState:
        raise NotImplementedError

    @abstractmethod
    def apply(self, action: Action) -> None:
        raise NotImplementedError
