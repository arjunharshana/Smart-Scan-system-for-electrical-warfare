from __future__ import annotations

from rf_environment.domain.enums import Serializable


class IdealSignal(Serializable):
    """Ideal transmitted signal before channel effects."""

    emitter_id: str
    timestamp: int
    frequency_hz: float
    bandwidth_hz: float
    power_dbm: float
    transmitting: bool
