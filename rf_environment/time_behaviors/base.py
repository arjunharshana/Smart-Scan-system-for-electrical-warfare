from __future__ import annotations

from abc import ABC, abstractmethod

from rf_environment.domain.enums import TimeBehaviorType


class TimeBehavior(ABC):
    """WHEN the emitter transmits."""

    behavior_type: TimeBehaviorType

    @abstractmethod
    def is_transmitting(self, time_step: int) -> bool:
        raise NotImplementedError
