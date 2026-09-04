from __future__ import annotations

from collections import deque
from typing import Any

from rf_environment.domain.metrics import MetricSnapshot
from rf_environment.metrics.opportunity_tracker import OpportunityTracker


class MetricsEngine:
    """Computes and tracks evaluation metrics for the RF simulation.

    Strictly separates:
    1. Detector performance (Pd, Pfa, Miss Rate, CR Rate)
    2. Scan strategy / Scheduler performance (Coverage, Detection Given Coverage, Interception Ratio, Delay)
    3. Learning & Prediction performance (Reward, Rolling Interception, Observed Transition Accuracy)
    """

    def __init__(self, rolling_window: int = 20) -> None:
        self.total_scans = 0
        self.hits = 0
        self.misses = 0
        self.false_alarms = 0
        self.correct_rejections = 0
        self.rewards: list[float] = []
        self.cumulative_reward = 0.0
        self.detected_emitters: set[str] = set()
        self.first_intercept: dict[str, int] = {}
        self.transmitting_emitter_steps = 0
        self.intercepted_emitter_steps = 0

        # Rolling window for interception rate
        self.rolling_window = rolling_window
        self._intercept_history: deque[int] = deque(maxlen=rolling_window)

        # Opportunity Tracker
        self.opportunity_tracker = OpportunityTracker()

        # Observed Transition Prediction Tracking
        self.total_predictions = 0
        self.correct_predictions = 0

    def record_prediction(self, correct: bool) -> None:
        """Records an observed frequency transition prediction outcome."""
        self.total_predictions += 1
        if correct:
            self.correct_predictions += 1

    def record(
        self,
        timestamp: int,
        outcome: str,
        reward: float,
        transmitting_ids: list[str],
        intercepted_ids: list[str],
        detected: bool,
        associated_emitter_id: str | None = None,
    ) -> MetricSnapshot:
        self.total_scans += 1
        self.rewards.append(reward)
        self.cumulative_reward += reward

        if outcome == "HIT":
            self.hits += 1
            self._intercept_history.append(1)
        elif outcome == "MISS":
            self.misses += 1
            self._intercept_history.append(0)
        elif outcome == "FALSE_ALARM":
            self.false_alarms += 1
            self._intercept_history.append(0)
        elif outcome == "CORRECT_REJECTION":
            self.correct_rejections += 1
            self._intercept_history.append(0)

        self.transmitting_emitter_steps += len(transmitting_ids)
        self.intercepted_emitter_steps += len(intercepted_ids)

        for emitter_id in intercepted_ids:
            self.detected_emitters.add(emitter_id)
            self.first_intercept.setdefault(emitter_id, timestamp)
        if associated_emitter_id and outcome == "HIT":
            self.detected_emitters.add(associated_emitter_id)
            self.first_intercept.setdefault(associated_emitter_id, timestamp)

        return self.snapshot(timestamp)

    def snapshot(self, timestamp: int) -> MetricSnapshot:
        in_band_trials = self.hits + self.misses
        empty_band_trials = self.false_alarms + self.correct_rejections
        pd = self.hits / in_band_trials if in_band_trials else 0.0
        pfa = self.false_alarms / empty_band_trials if empty_band_trials else 0.0
        miss_rate = self.misses / in_band_trials if in_band_trials else 0.0
        cr_rate = self.correct_rejections / empty_band_trials if empty_band_trials else 0.0

        opp_summary = self.opportunity_tracker.compute_summary()

        step_coverage_ratio = (
            self.intercepted_emitter_steps / self.transmitting_emitter_steps
            if self.transmitting_emitter_steps
            else 0.0
        )
        avg_reward = self.cumulative_reward / self.total_scans if self.total_scans else 0.0
        rolling_intercept = (
            sum(self._intercept_history) / len(self._intercept_history)
            if self._intercept_history
            else 0.0
        )
        pred_acc = (
            self.correct_predictions / self.total_predictions
            if self.total_predictions
            else 0.0
        )

        return MetricSnapshot(
            timestamp=timestamp,
            total_scans=self.total_scans,
            # Detector
            hits=self.hits,
            misses=self.misses,
            false_alarms=self.false_alarms,
            correct_rejections=self.correct_rejections,
            probability_of_detection=pd,
            probability_of_false_alarm=pfa,
            miss_rate=miss_rate,
            correct_rejection_rate=cr_rate,
            # Scheduler
            metric_scope=opp_summary.get("metric_scope", "HOP"),
            opportunities_total=opp_summary["opportunities_total"],
            opportunities_covered=opp_summary["opportunities_covered"],
            opportunities_intercepted=opp_summary["opportunities_intercepted"],
            opportunity_coverage=opp_summary["opportunity_coverage"],
            detection_given_coverage=opp_summary["detection_given_coverage"],
            interception_ratio=opp_summary["interception_ratio"],
            # Explicit Hop-Level Metrics
            total_hops=opp_summary.get("total_hops", 0),
            covered_hops=opp_summary.get("covered_hops", 0),
            intercepted_hops=opp_summary.get("intercepted_hops", 0),
            hop_coverage_ratio=opp_summary.get("hop_coverage_ratio", 0.0),
            hop_detection_ratio=opp_summary.get("hop_detection_ratio", 0.0),
            hop_interception_ratio=opp_summary.get("hop_interception_ratio", 0.0),
            # Macro Episode Metrics
            total_episodes=opp_summary.get("total_episodes", 0),
            covered_episodes=opp_summary.get("covered_episodes", 0),
            intercepted_episodes=opp_summary.get("intercepted_episodes", 0),
            episode_coverage_ratio=opp_summary.get("episode_coverage_ratio", 0.0),
            episode_detection_given_coverage=opp_summary.get("episode_detection_given_coverage", 0.0),
            episode_interception_ratio=opp_summary.get("episode_interception_ratio", 0.0),
            average_intercept_time=opp_summary["average_intercept_time"],
            median_intercept_time=opp_summary["median_intercept_time"],
            intercept_time_std=opp_summary["intercept_time_std"],
            unique_emitters_detected=len(self.detected_emitters),
            time_to_first_intercept=dict(self.first_intercept),
            transmitting_emitter_steps=self.transmitting_emitter_steps,
            intercepted_emitter_steps=self.intercepted_emitter_steps,
            step_coverage_ratio=step_coverage_ratio,
            identity_verified=opp_summary.get("identity_verified", True),
            # Learning & Prediction
            cumulative_reward=self.cumulative_reward,
            average_reward=avg_reward,
            rolling_interception_rate=rolling_intercept,
            prediction_accuracy=pred_acc,
            total_predictions=self.total_predictions,
            correct_predictions=self.correct_predictions,
            # Compatibility
            average_intercept_rate=opp_summary["interception_ratio"],
            average_intercept_time_error=opp_summary["intercept_time_std"],
        )

    def reset(self) -> None:
        self.__init__(rolling_window=self.rolling_window)
