from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.metrics import SchedulerTelemetryState
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler


class ContextAwareScheduler(BaseScheduler):
    """Simplified 3-term Context-Aware / Transition Scan Scheduler.

    Combines three interpretable terms using configurable weights:
    1. Transition Probability: P(next_band | last_detected_band) learned purely
       from observed receiver detections.
    2. Recent Activity: Frequency of detections in recent scans of each band.
    3. Exploration / Coverage: Bonus for bands that have not been probed recently.

    Total Score(j) = w_trans * Transition(j) + w_act * Activity(j) + w_exp * Exploration(j)
    """

    name = "context_aware"
    category = "contextual"

    def __init__(
        self,
        bands_hz: list[float],
        w_transition: float = 0.55,
        w_activity: float = 0.25,
        w_exploration: float = 0.20,
        smoothing: float = 0.05,
        stale_threshold: int = 15,
        seed: int | None = None,
    ) -> None:
        super().__init__(bands_hz)
        self.w_trans = float(w_transition)
        self.w_act = float(w_activity)
        self.w_exp = float(w_exploration)
        self.smoothing = float(smoothing)
        self.stale_threshold = int(stale_threshold)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        n = len(self.bands_hz)
        # Empirical transition counts: src_bin -> dst_bin -> count
        self.counts_by_bin: list[list[int]] = [[0] * n for _ in range(n)]
        self.last_detected_bin: int | None = None
        self.last_scanned_time: list[int] = [-999] * n
        self.t = 0
        self.predicted_frequency_hz: float | None = None

    @property
    def counts(self) -> dict[float, dict[float, int]]:
        """Legacy compatibility mapping of counts by Hz."""
        return {
            src_f: {dst_f: self.counts_by_bin[i][j] for j, dst_f in enumerate(self.bands_hz)}
            for i, src_f in enumerate(self.bands_hz)
        }

    @property
    def last_detected_freq(self) -> float | None:
        return self.bands_hz[self.last_detected_bin] if self.last_detected_bin is not None else None

    def _closest_band_idx(self, freq_hz: float) -> int:
        return min(range(len(self.bands_hz)), key=lambda i: abs(self.bands_hz[i] - freq_hz))

    def compute_action_scores(
        self, observation: SchedulerObservation | Any = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Computes the full vector of Context-Aware scores, transition probabilities, and activity levels."""
        n = len(self.bands_hz)

        # 1. Determine last detected bin
        last_det_bin = None
        if isinstance(observation, SchedulerObservation):
            last_det_bin = observation.last_detection_bin
        if last_det_bin is None:
            last_det_bin = self.last_detected_bin

        # 2. Transition probabilities T_j = P(bin_j | last_det_bin)
        transition_probs = [1.0 / n] * n
        if last_det_bin is not None and 0 <= last_det_bin < n:
            row = self.counts_by_bin[last_det_bin]
            total_pulls = sum(row)
            denom = total_pulls + self.smoothing * n
            transition_probs = [(row[j] + self.smoothing) / denom for j in range(n)]

        best_dst_idx = max(range(n), key=lambda j: transition_probs[j])
        self.predicted_frequency_hz = self.bands_hz[best_dst_idx]

        # 3. Activity levels A_j from recent history: detection rate of bin_j when probed
        activity_scores = [0.0] * n
        if isinstance(observation, SchedulerObservation) and observation.recent_detection_history:
            for j in range(n):
                scans_j = sum(1 for b in observation.recent_frequency_history if b == j)
                if scans_j > 0:
                    dets_j = sum(
                        1
                        for det, b in zip(
                            observation.recent_detection_history,
                            observation.recent_frequency_history,
                        )
                        if det and b == j
                    )
                    activity_scores[j] = dets_j / scans_j

        # 4. Exploration scores E_j
        exploration_scores = [1.0] * n
        if isinstance(observation, SchedulerObservation) and observation.time_since_scan_by_bin:
            stale = max(self.stale_threshold, 1)
            exploration_scores = [
                min(1.0, observation.time_since_scan_by_bin[j] / stale)
                for j in range(n)
            ]
        else:
            for j in range(n):
                elapsed = self.t - self.last_scanned_time[j]
                exploration_scores[j] = min(1.0, elapsed / max(self.stale_threshold, 1))

        # 5. Combined Score
        scores = np.zeros(n, dtype=np.float32)
        for j in range(n):
            scores[j] = (
                self.w_trans * transition_probs[j]
                + self.w_act * activity_scores[j]
                + self.w_exp * exploration_scores[j]
            )

        details = {
            "transition_probs": np.array(transition_probs, dtype=np.float32),
            "activity_scores": np.array(activity_scores, dtype=np.float32),
            "exploration_scores": np.array(exploration_scores, dtype=np.float32),
            "last_detected_bin": last_det_bin,
            "predicted_bin": best_dst_idx,
        }
        return scores, details

    def select_bin(self, observation: SchedulerObservation | Any = None) -> int:
        self.t += 1
        n = len(self.bands_hz)
        scores, details = self.compute_action_scores(observation)
        transition_probs = details["transition_probs"]
        activity_scores = details["activity_scores"]
        exploration_scores = details["exploration_scores"]
        last_det_bin = details["last_detected_bin"]

        # Break ties with small deterministic jitter
        jitter = [float(self.rng.uniform(0.0, 1e-6)) for _ in range(n)]
        chosen_bin = int(max(range(n), key=lambda j: scores[j] + jitter[j]))

        self.last_selected_bin = chosen_bin
        self.last_selected = self.bands_hz[chosen_bin]
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "reason": (
                f"Combined score {scores[chosen_bin]:.3f} = "
                f"{self.w_trans:.2f}×Trans({transition_probs[chosen_bin]:.2f} from {self.bands_hz[last_det_bin]/1e6 if last_det_bin is not None else 'None'}) + "
                f"{self.w_act:.2f}×Act({activity_scores[chosen_bin]:.2f}) + "
                f"{self.w_exp:.2f}×Exp({exploration_scores[chosen_bin]:.2f})"
            ),
            "rule": "weighted_context_fusion",
            "transition_probability": float(transition_probs[chosen_bin]),
            "activity_level": float(activity_scores[chosen_bin]),
            "exploration_bonus": float(exploration_scores[chosen_bin]),
            "estimated_value": float(scores[chosen_bin]),
            "predicted_next_mhz": self.predicted_frequency_hz / 1e6 if self.predicted_frequency_hz else None,
        }
        return chosen_bin

    def select_frequency(self, observation: Any = None) -> float:
        bin_idx = self.select_bin(observation)
        return self.bands_hz[bin_idx]

    def update_policy(
        self,
        observation: SchedulerObservation,
        action: ScanAction,
        reward: float,
        next_observation: SchedulerObservation,
        done: bool,
    ) -> None:
        scanned_bin = action.frequency_bin
        if 0 <= scanned_bin < len(self.last_scanned_time):
            self.last_scanned_time[scanned_bin] = self.t

        if next_observation.last_detection:
            if self.last_detected_bin is not None and 0 <= self.last_detected_bin < len(self.bands_hz):
                self.counts_by_bin[self.last_detected_bin][scanned_bin] += 1
            self.last_detected_bin = scanned_bin

    def observe(
        self,
        observation: SchedulerObservation | Any,
        action: ScanAction | None = None,
        reward: float = 0.0,
        next_observation: SchedulerObservation | None = None,
        done: bool = False,
    ) -> None:
        if hasattr(observation, "observation") and hasattr(observation, "action") and hasattr(observation, "reward"):
            trans = observation
            self.update_policy(trans.observation, trans.action, trans.reward, trans.next_observation, trans.done)
            return
        self.update_policy(observation, action, reward, next_observation, done)

    def update(self, observation: Any, reward: float, action: float | int | None = None) -> None:
        if action is not None:
            if isinstance(action, int):
                scanned_bin = action
            else:
                scanned_bin = self._closest_band_idx(float(action))
        else:
            scanned_bin = self.last_selected_bin

        if 0 <= scanned_bin < len(self.last_scanned_time):
            self.last_scanned_time[scanned_bin] = self.t

        detected = getattr(observation, "detected", False) or getattr(observation, "last_detection", False)
        if detected:
            if self.last_detected_bin is not None and 0 <= self.last_detected_bin < len(self.bands_hz):
                self.counts_by_bin[self.last_detected_bin][scanned_bin] += 1
            self.last_detected_bin = scanned_bin

    def reset(self) -> None:
        super().reset()
        n = len(self.bands_hz)
        self.counts_by_bin = [[0] * n for _ in range(n)]
        self.last_detected_bin = None
        self.last_scanned_time = [-999] * n
        self.t = 0
        self.predicted_frequency_hz = None
        self.rng = np.random.default_rng(self.seed)

    def get_state(self) -> SchedulerTelemetryState:
        stats = []
        n = len(self.bands_hz)
        for i, b in enumerate(self.bands_hz):
            row = self.counts_by_bin[i]
            total_pulls = sum(row)
            stats.append({
                "frequency_hz": b,
                "count": total_pulls,
                "value": total_pulls,
                "probability": 1.0 / n,
                "transitions_from": {f"{dst/1e6:.1f}": row[j] for j, dst in enumerate(self.bands_hz)},
            })
        return SchedulerTelemetryState(
            name=self.name,
            category=self.category,
            selected_frequency_hz=self.last_selected,
            arm_stats=stats,
            explanation=self.get_explanation(),
        )
