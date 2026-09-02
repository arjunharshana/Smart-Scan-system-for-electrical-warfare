from __future__ import annotations

from rf_environment.domain.enums import Serializable
from rf_environment.domain.receiver import ReceiverMeasurement


class DetectionResult(Serializable):
    detected: bool
    timestamp: int
    frequency_hz: float
    snr_db: float | None = None
    emitter_id: str | None = None
    false_alarm: bool = False
    associated: bool = False


class Observation(Serializable):
    """What the scheduler is allowed to see. Never includes ground truth."""

    timestamp: int
    receiver_frequency_hz: float
    receiver_bandwidth_hz: float
    detected: bool
    snr_db: float | None = None
    associated_emitter_id: str | None = None
    reward: float | None = None
    measurement: ReceiverMeasurement | None = None
