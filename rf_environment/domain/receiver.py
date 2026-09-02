from __future__ import annotations

from rf_environment.domain.enums import Serializable


class ReceiverState(Serializable):
    center_frequency_hz: float
    instantaneous_bandwidth_hz: float
    sensitivity_dbm: float
    noise_floor_dbm: float
    detection_threshold_db: float
    tuning_time_ms: float
    remaining_tune_steps: int = 0
    status: str = "IDLE"


class ReceiverMeasurement(Serializable):
    timestamp: int
    center_frequency_hz: float
    bandwidth_hz: float
    in_band_emitter_ids: list[str] = []
    signal_power_dbm: float | None = None
    noise_power_dbm: float = -100.0
    snr_db: float | None = None
    tuning: bool = False
