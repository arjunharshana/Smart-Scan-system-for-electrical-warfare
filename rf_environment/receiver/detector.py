from __future__ import annotations

import numpy as np

from rf_environment.domain.observation import DetectionResult
from rf_environment.domain.receiver import ReceiverMeasurement


class Detector:
    """Imperfect energy detector with Pd and Pfa."""

    def __init__(
        self,
        detection_threshold_db: float = 6.0,
        p_detection: float = 0.9,
        p_false_alarm: float = 0.02,
        seed: int | None = None,
    ) -> None:
        self.detection_threshold_db = float(detection_threshold_db)
        self.p_detection = float(p_detection)
        self.p_false_alarm = float(p_false_alarm)
        self.rng = np.random.default_rng(seed)

    def detect(self, measurement: ReceiverMeasurement) -> DetectionResult:
        if measurement.tuning:
            return DetectionResult(
                detected=False,
                timestamp=measurement.timestamp,
                frequency_hz=measurement.center_frequency_hz,
                snr_db=measurement.snr_db,
            )
        signal_present = measurement.snr_db is not None and measurement.snr_db >= self.detection_threshold_db
        if signal_present:
            detected = bool(self.rng.random() < self.p_detection)
            emitter_id = measurement.in_band_emitter_ids[0] if measurement.in_band_emitter_ids else None
            return DetectionResult(
                detected=detected,
                timestamp=measurement.timestamp,
                frequency_hz=measurement.center_frequency_hz,
                snr_db=measurement.snr_db,
                emitter_id=emitter_id if detected else None,
                associated=bool(detected and emitter_id),
                false_alarm=False,
            )
        false_alarm = bool(self.rng.random() < self.p_false_alarm)
        return DetectionResult(
            detected=false_alarm,
            timestamp=measurement.timestamp,
            frequency_hz=measurement.center_frequency_hz,
            snr_db=measurement.snr_db,
            false_alarm=false_alarm,
        )
