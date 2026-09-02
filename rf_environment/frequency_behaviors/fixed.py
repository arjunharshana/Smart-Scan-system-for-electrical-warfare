from __future__ import annotations

from rf_environment.domain.enums import FrequencyBehaviorType
from rf_environment.frequency_behaviors.base import FrequencyBehavior


class FixedFrequency(FrequencyBehavior):
    behavior_type = FrequencyBehaviorType.FIXED

    def __init__(self, frequency_hz: float) -> None:
        self.frequency_hz = float(frequency_hz)

    def get_frequency(self, time_step: int, state: dict | None = None) -> float:
        return self.frequency_hz
