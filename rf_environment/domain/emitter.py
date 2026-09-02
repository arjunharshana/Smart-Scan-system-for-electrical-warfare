from __future__ import annotations

from rf_environment.domain.enums import (
    EmitterType,
    FrequencyBehaviorType,
    Serializable,
    TimeBehaviorType,
)


class EmitterState(Serializable):
    emitter_id: str
    emitter_type: EmitterType
    timestamp: int
    transmitting: bool
    frequency_hz: float
    bandwidth_hz: float
    power_dbm: float
    frequency_behavior: FrequencyBehaviorType
    time_behavior: TimeBehaviorType
    modulation: str = "UNKNOWN"
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
