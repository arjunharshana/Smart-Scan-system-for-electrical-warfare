from __future__ import annotations

from rf_environment.domain.enums import Serializable


class SchedulerState(Serializable):
    name: str
    selected_frequency_hz: float | None = None
    arm_stats: list[dict] = []


class MetricSnapshot(Serializable):
    timestamp: int
    total_scans: int = 0
    hits: int = 0
    misses: int = 0
    false_alarms: int = 0
    correct_rejections: int = 0
    unique_emitters_detected: int = 0
    probability_of_detection: float = 0.0
    probability_of_false_alarm: float = 0.0
    interception_ratio: float = 0.0
    average_intercept_rate: float = 0.0
    average_reward: float = 0.0
    average_intercept_time: float | None = None
    prediction_accuracy: float = 0.0
    average_intercept_time_error: float | None = None
    time_to_first_intercept: dict[str, int | None] = {}
    transmitting_emitter_steps: int = 0
    intercepted_emitter_steps: int = 0
