from __future__ import annotations

from rf_environment.domain.metrics import MetricSnapshot


class MetricsEngine:
    def __init__(self) -> None:
        self.total_scans = 0
        self.hits = 0
        self.misses = 0
        self.false_alarms = 0
        self.correct_rejections = 0
        self.rewards: list[float] = []
        self.detected_emitters: set[str] = set()
        self.first_intercept: dict[str, int] = {}
        self.transmitting_emitter_steps = 0
        self.intercepted_emitter_steps = 0
        self.correct_predictions = 0

    def record(
        self,
        timestamp: int,
        outcome: str,
        reward: float,
        transmitting_ids: list[str],
        intercepted_ids: list[str],
        detected: bool,
        associated_emitter_id: str | None,
    ) -> MetricSnapshot:
        self.total_scans += 1
        self.rewards.append(reward)
        if outcome == "HIT":
            self.hits += 1
            self.correct_predictions += 1
        elif outcome == "MISS":
            self.misses += 1
        elif outcome == "FALSE_ALARM":
            self.false_alarms += 1
        elif outcome == "CORRECT_REJECTION":
            self.correct_rejections += 1
            self.correct_predictions += 1

        self.transmitting_emitter_steps += len(transmitting_ids)
        self.intercepted_emitter_steps += len(intercepted_ids)
        for emitter_id in intercepted_ids:
            self.detected_emitters.add(emitter_id)
            self.first_intercept.setdefault(emitter_id, timestamp)
        if associated_emitter_id:
            self.detected_emitters.add(associated_emitter_id)
            self.first_intercept.setdefault(associated_emitter_id, timestamp)

        return self.snapshot(timestamp)

    def snapshot(self, timestamp: int) -> MetricSnapshot:
        in_band_trials = self.hits + self.misses
        empty_band_trials = self.false_alarms + self.correct_rejections
        pd = self.hits / in_band_trials if in_band_trials else 0.0
        pfa = self.false_alarms / empty_band_trials if empty_band_trials else 0.0
        intercept_ratio = (
            self.intercepted_emitter_steps / self.transmitting_emitter_steps
            if self.transmitting_emitter_steps
            else 0.0
        )
        avg_reward = sum(self.rewards) / len(self.rewards) if self.rewards else 0.0
        intercept_times = list(self.first_intercept.values())
        avg_intercept_time = sum(intercept_times) / len(intercept_times) if intercept_times else None
        accuracy = self.correct_predictions / self.total_scans if self.total_scans else 0.0
        return MetricSnapshot(
            timestamp=timestamp,
            total_scans=self.total_scans,
            hits=self.hits,
            misses=self.misses,
            false_alarms=self.false_alarms,
            correct_rejections=self.correct_rejections,
            unique_emitters_detected=len(self.detected_emitters),
            probability_of_detection=pd,
            probability_of_false_alarm=pfa,
            interception_ratio=intercept_ratio,
            average_intercept_rate=intercept_ratio,
            average_reward=avg_reward,
            average_intercept_time=avg_intercept_time,
            prediction_accuracy=accuracy,
            average_intercept_time_error=None,
            time_to_first_intercept=dict(self.first_intercept),
            transmitting_emitter_steps=self.transmitting_emitter_steps,
            intercepted_emitter_steps=self.intercepted_emitter_steps,
        )

    def reset(self) -> None:
        self.__init__()
