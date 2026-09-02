from __future__ import annotations

from rf_environment.domain.enums import Serializable, TimeBehaviorType


class TimeSnapshot(Serializable):
    behavior: TimeBehaviorType
    transmitting: bool
