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


class TemporalHistory(Serializable):
    """Recent receiver observation history without ground truth."""

    previous_frequencies: list[float] = []
    previous_detections: list[bool] = []
    previous_rewards: list[float] = []
    last_detected_frequency_hz: float | None = None
    time_since_last_detection: int | None = None
    band_activity_levels: dict[str, float] = {}
    time_since_band_last_scanned: dict[str, int] = {}
    recent_transitions: list[list[float]] = []


class Observation(Serializable):
    """What the scheduler is allowed to see. Never includes ground truth."""

    timestamp: int
    receiver_frequency_hz: float
    receiver_bandwidth_hz: float
    detected: bool
    snr_db: float | None = None
    associated_emitter_id: str | None = None
    reward: float | None = None
    previous_action: float | None = None
    recent_history: TemporalHistory | None = None
    measurement: ReceiverMeasurement | None = None
