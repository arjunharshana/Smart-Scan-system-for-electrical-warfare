"""Shared JSON-serializable helpers."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class Serializable(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class EmitterType(str, Enum):
    RADAR = "RADAR"
    COMMUNICATION = "COMMUNICATION"


class FrequencyBehaviorType(str, Enum):
    FIXED = "FIXED"
    HOPPING = "HOPPING"
    SWEEP = "SWEEP"
    RANDOM = "RANDOM"


class TimeBehaviorType(str, Enum):
    CONTINUOUS = "CONTINUOUS"
    PERIODIC = "PERIODIC"
    BURST = "BURST"
    INTERMITTENT = "INTERMITTENT"


class EventType(str, Enum):
    SIMULATION_STARTED = "SIMULATION_STARTED"
    SIMULATION_PAUSED = "SIMULATION_PAUSED"
    SIMULATION_STEP = "SIMULATION_STEP"
    EMITTER_STATE_CHANGED = "EMITTER_STATE_CHANGED"
    SCAN_STARTED = "SCAN_STARTED"
    SCAN_RESULT = "SCAN_RESULT"
    HIT = "HIT"
    MISS = "MISS"
    FALSE_ALARM = "FALSE_ALARM"
    SCHEDULER_DECISION = "SCHEDULER_DECISION"
    METRIC_UPDATE = "METRIC_UPDATE"
    SIMULATION_COMPLETED = "SIMULATION_COMPLETED"
    SIMULATION_RESET = "SIMULATION_RESET"
    SIMULATION_STOPPED = "SIMULATION_STOPPED"


class OutcomeType(str, Enum):
    HIT = "HIT"
    MISS = "MISS"
    FALSE_ALARM = "FALSE_ALARM"
    CORRECT_REJECTION = "CORRECT_REJECTION"
