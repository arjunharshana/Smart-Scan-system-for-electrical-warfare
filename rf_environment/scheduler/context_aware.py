from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.domain.metrics import SchedulerState
from rf_environment.domain.observation import Observation
from rf_environment.scheduler.base import ScanScheduler


class ContextAwareScheduler(ScanScheduler):
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

        # Empirical transition counts from observed detections: src -> dst -> count
        self.counts: dict[float, dict[float, int]] = {
            src: {dst: 0 for dst in self.bands_hz} for src in self.bands_hz
        }
        self.last_detected_freq: float | None = None
        self.last_scanned_time: dict[float, int] = {b: -999 for b in self.bands_hz}
        self.t = 0
        self.predicted_frequency_hz: float | None = None

    def _closest_band(self, freq_hz: float) -> float:
        return min(self.bands_hz, key=lambda b: abs(b - freq_hz))

    def select_frequency(self, observation: Observation | None) -> float:
        self.t += 1
        n = len(self.bands_hz)

        # 1. Determine last detected frequency from observation context
        last_det = None
        if observation and observation.recent_history:
            last_det = observation.recent_history.last_detected_frequency_hz
        if last_det is None:
            last_det = self.last_detected_freq

        # 2. Compute Transition Probabilities T_j = P(band_j | last_det)
        transition_probs = {}
        if last_det is not None:
            src = self._closest_band(last_det)
            row = self.counts[src]
            total_pulls = sum(row.values())
            denom = total_pulls + self.smoothing * n
            for dst in self.bands_hz:
                transition_probs[dst] = (row[dst] + self.smoothing) / denom
        else:
            for dst in self.bands_hz:
                transition_probs[dst] = 1.0 / n

        # Next frequency prediction is the argmax of transition probability
        self.predicted_frequency_hz = max(self.bands_hz, key=lambda b: transition_probs[b])

        # 3. Compute Activity Levels A_j
        activity_levels = {}
        if observation and observation.recent_history and observation.recent_history.band_activity_levels:
            for b in self.bands_hz:
                key = f"{b/1e6:.1f}"
                activity_levels[b] = observation.recent_history.band_activity_levels.get(key, 0.0)
        else:
            activity_levels = {b: 0.0 for b in self.bands_hz}

        # 4. Compute Exploration / Coverage term E_j based on elapsed idle time
        exploration_scores = {}
        for b in self.bands_hz:
            elapsed = self.t - self.last_scanned_time.get(b, -999)
            exploration_scores[b] = min(1.0, elapsed / max(self.stale_threshold, 1))

        # 5. Combined Score
        scores = {}
        for b in self.bands_hz:
            s = (
                self.w_trans * transition_probs[b]
                + self.w_act * activity_levels[b]
                + self.w_exp * exploration_scores[b]
            )
            # Small random tie-breaker
            scores[b] = s + self.rng.uniform(0, 1e-6)

        best_band = max(self.bands_hz, key=lambda b: scores[b])
        self.last_selected = best_band
        self.last_scanned_time[best_band] = self.t

        t_prob = transition_probs[best_band]
        act_score = activity_levels[best_band]
        exp_score = exploration_scores[best_band]
        tot = scores[best_band]

        prev_str = f"{last_det/1e6:.0f} MHz" if last_det is not None else "None"
        self.last_explanation = {
            "action_mhz": best_band / 1e6,
            "reason": (
                f"Combined score {tot:.3f} = {self.w_trans:.2f}×Trans({t_prob:.2f} from {prev_str}) + "
                f"{self.w_act:.2f}×Act({act_score:.2f}) + {self.w_exp:.2f}×Exp({exp_score:.2f})"
            ),
            "rule": "weighted_context_fusion",
            "transition_probability": t_prob,
            "activity_level": act_score,
            "exploration_bonus": exp_score,
            "estimated_value": tot,
            "predicted_next_mhz": self.predicted_frequency_hz / 1e6 if self.predicted_frequency_hz else None,
        }
        return self.last_selected

    def update(self, observation: Observation, reward: float, action: float | None = None) -> None:
        scanned_freq = action or observation.receiver_frequency_hz
        band = self._closest_band(scanned_freq)
        self.last_scanned_time[band] = self.t

        if observation.detected:
            if self.last_detected_freq is not None:
                src = self._closest_band(self.last_detected_freq)
                dst = self._closest_band(scanned_freq)
                self.counts[src][dst] += 1
            self.last_detected_freq = scanned_freq

    def reset(self) -> None:
        super().reset()
        self.counts = {src: {dst: 0 for dst in self.bands_hz} for src in self.bands_hz}
        self.last_detected_freq = None
        self.last_scanned_time = {b: -999 for b in self.bands_hz}
        self.t = 0
        self.predicted_frequency_hz = None
        self.rng = np.random.default_rng(self.seed)

    def get_state(self) -> SchedulerState:
        stats = []
        n = len(self.bands_hz)
        for b in self.bands_hz:
            row = self.counts[b]
            total_pulls = sum(row.values())
            stats.append({
                "frequency_hz": b,
                "count": total_pulls,
                "value": total_pulls,
                "probability": 1.0 / n,
                "transitions_from": dict(row),
            })
        return SchedulerState(
            name=self.name,
            category=self.category,
            selected_frequency_hz=self.last_selected,
            arm_stats=stats,
            explanation=self.get_explanation(),
        )
