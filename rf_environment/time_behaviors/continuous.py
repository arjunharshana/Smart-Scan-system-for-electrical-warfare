from __future__ import annotations

from rf_environment.domain.enums import TimeBehaviorType
from rf_environment.time_behaviors.base import TimeBehavior


class Continuous(TimeBehavior):
    behavior_type = TimeBehaviorType.CONTINUOUS

    def is_transmitting(self, time_step: int) -> bool:
        return True
