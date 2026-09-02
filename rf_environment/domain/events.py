from __future__ import annotations

from rf_environment.domain.enums import EventType, Serializable


class SimulationEvent(Serializable):
    timestamp: int
    event_type: EventType
    payload: dict = {}
