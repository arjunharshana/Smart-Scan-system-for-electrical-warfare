from __future__ import annotations

from rf_environment.domain.enums import TimeBehaviorType
from rf_environment.time_behaviors.base import TimeBehavior


class Periodic(TimeBehavior):
    behavior_type = TimeBehaviorType.PERIODIC

    def __init__(self, on_duration: int, off_duration: int, phase: int = 0) -> None:
        if on_duration < 0 or off_duration < 0:
            raise ValueError("durations must be non-negative")
        if on_duration + off_duration <= 0:
            raise ValueError("period must be positive")
        self.on_duration = int(on_duration)
        self.off_duration = int(off_duration)
        self.phase = int(phase)
        self.period = self.on_duration + self.off_duration

    def is_transmitting(self, time_step: int) -> bool:
        t = (time_step + self.phase) % self.period
        return t < self.on_duration
