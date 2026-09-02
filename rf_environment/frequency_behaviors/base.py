from __future__ import annotations

from abc import ABC, abstractmethod

from rf_environment.domain.enums import FrequencyBehaviorType


class FrequencyBehavior(ABC):
    """WHERE the emitter transmits."""

    behavior_type: FrequencyBehaviorType

    @abstractmethod
    def get_frequency(self, time_step: int, state: dict | None = None) -> float:
        raise NotImplementedError
