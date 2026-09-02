from __future__ import annotations

from rf_environment.domain.enums import TimeBehaviorType
from rf_environment.time_behaviors.base import TimeBehavior


class Burst(TimeBehavior):
    behavior_type = TimeBehaviorType.BURST

    def __init__(self, burst_duration: int, interval: int, start_time: int = 0) -> None:
        if burst_duration <= 0 or interval <= 0:
            raise ValueError("burst_duration and interval must be positive")
        if burst_duration > interval:
            raise ValueError("burst_duration cannot exceed interval")
        self.burst_duration = int(burst_duration)
        self.interval = int(interval)
        self.start_time = int(start_time)

    def is_transmitting(self, time_step: int) -> bool:
        if time_step < self.start_time:
            return False
        offset = (time_step - self.start_time) % self.interval
        return offset < self.burst_duration
