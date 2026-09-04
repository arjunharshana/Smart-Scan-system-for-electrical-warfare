from __future__ import annotations

from rf_environment.domain.enums import Serializable


class SchedulerState(Serializable):
    name: str
    category: str = "baseline"  # baseline, non_stationary, contextual
    selected_frequency_hz: float | None = None
    arm_stats: list[dict] = []
    explanation: dict = {}


class MetricSnapshot(Serializable):
    """Snapshot of simulation metrics with strict mathematical separation.

    Separates:
    1. Detector performance (Pd, Pfa, Miss Rate, CR Rate)
    2. Scheduler performance (Coverage, Detection Given Coverage, Interception Ratio, Delay)
    3. Learning performance (Reward, Rolling Interception, Prediction Accuracy)
    """

    timestamp: int
    total_scans: int = 0

    # 1. Detector Performance
    hits: int = 0
    misses: int = 0
    false_alarms: int = 0
    correct_rejections: int = 0
    probability_of_detection: float = 0.0
    probability_of_false_alarm: float = 0.0
    miss_rate: float = 0.0
    correct_rejection_rate: float = 0.0

    # 2. Scheduler Performance (Opportunity-Based)
    metric_scope: str = "HOP"  # HOP, BURST, EPISODE
    opportunities_total: int = 0
    opportunities_covered: int = 0
    opportunities_intercepted: int = 0
    opportunity_coverage: float = 0.0
    detection_given_coverage: float = 0.0
    interception_ratio: float = 0.0

    # Explicit Hop-Level Metrics
    total_hops: int = 0
    covered_hops: int = 0
    intercepted_hops: int = 0
    hop_coverage_ratio: float = 0.0
    hop_detection_ratio: float = 0.0
    hop_interception_ratio: float = 0.0

    # Macro Episode Metrics
    total_episodes: int = 0
    covered_episodes: int = 0
    intercepted_episodes: int = 0
    episode_coverage_ratio: float = 0.0
    episode_detection_given_coverage: float = 0.0
    episode_interception_ratio: float = 0.0

    average_intercept_time: float | None = None
    median_intercept_time: float | None = None
    intercept_time_std: float | None = None
    unique_emitters_detected: int = 0
    time_to_first_intercept: dict[str, int | None] = {}
    transmitting_emitter_steps: int = 0
    intercepted_emitter_steps: int = 0
    step_coverage_ratio: float = 0.0
    identity_verified: bool = True

    # 3. Learning & Prediction Performance
    cumulative_reward: float = 0.0
    average_reward: float = 0.0
    rolling_interception_rate: float = 0.0
    prediction_accuracy: float = 0.0
    total_predictions: int = 0
    correct_predictions: int = 0

    # Legacy/Compatibility aliases
    average_intercept_rate: float = 0.0
    average_intercept_time_error: float | None = None
