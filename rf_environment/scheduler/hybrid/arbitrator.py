from __future__ import annotations

from enum import Enum
from typing import Any
import numpy as np

from rf_environment.domain.state import SchedulerObservation


class ArbitrationMode(str, Enum):
    DDQN_EXPLOIT = "DDQN_EXPLOIT"
    CA_ADAPT = "CA_ADAPT"
    BLENDED = "BLENDED"
    EXPLORE_DISCOVERY = "EXPLORE_DISCOVERY"


class HybridMetaArbitrator:
    """Transparent, deterministic rule-based Meta-Arbitrator for V4.0 Hybrid Scheduler.

    Arbitrates between the predictive DDQN branch and the adaptive Context-Aware branch
    using strictly legitimate receiver observables (zero ground truth).

    Signals Evaluated:
    1. DDQN Confidence: Normalized separation between top-2 Q-values.
    2. CA Confidence: Normalized transition probability peak above uniform and activity level.
    3. Surprise / Novelty: Mismatch between prediction and detector feedback (e.g. miss on
       high-confidence channel or detection on an unpredicted channel).
    4. Transition Consistency: Regularity and repeatability of recent observed transitions.
    5. Recent Performance: Rolling detection rate over recent window.
    """

    def __init__(
        self,
        num_bins: int,
        ddqn_conf_threshold: float = 0.25,
        ca_conf_threshold: float = 0.30,
        surprise_threshold: float = 0.40,
        consistency_threshold: float = 0.40,
        surprise_decay: float = 0.85,
        default_ddqn_weight: float = 0.70,
        default_ca_weight: float = 0.30,
    ) -> None:
        self.num_bins = int(num_bins)
        self.ddqn_conf_threshold = float(ddqn_conf_threshold)
        self.ca_conf_threshold = float(ca_conf_threshold)
        self.surprise_threshold = float(surprise_threshold)
        self.consistency_threshold = float(consistency_threshold)
        self.surprise_decay = float(surprise_decay)
        self.default_ddqn_weight = float(default_ddqn_weight)
        self.default_ca_weight = float(default_ca_weight)

        # Dynamic observable state tracking (reset on new episode)
        self.surprise: float = 0.0
        self.consecutive_misses_on_expected: int = 0
        self.last_predicted_bin: int | None = None
        self.last_chosen_branch: str = "BLENDED"
        self.last_decision_telemetry: dict[str, Any] = {}

    def reset(self) -> None:
        """Resets dynamic arbitration state at episode boundary."""
        self.surprise = 0.0
        self.consecutive_misses_on_expected = 0
        self.last_predicted_bin = None
        self.last_chosen_branch = "BLENDED"
        self.last_decision_telemetry.clear()

    def update_feedback(
        self,
        observation: SchedulerObservation,
        selected_bin: int,
        next_observation: SchedulerObservation,
    ) -> None:
        """Updates internal surprise and performance trackers after scan execution."""
        detected = bool(next_observation.last_detection)
        instant_surprise = 0.0

        # 1. If we scanned the predicted bin and got a miss
        if self.last_predicted_bin is not None and selected_bin == self.last_predicted_bin:
            if not detected:
                self.consecutive_misses_on_expected += 1
                # 1 miss is normal in hopping; 2+ consecutive misses indicate potential break
                if self.consecutive_misses_on_expected > 1:
                    instant_surprise = min(1.0, 0.40 * (self.consecutive_misses_on_expected - 1))
            else:
                self.consecutive_misses_on_expected = 0
                instant_surprise = 0.0
                # Fast recovery on confirmed detection
                self.surprise *= 0.60

        # 2. If a detection occurred on an unpredicted / novel bin
        if detected and next_observation.last_detection_bin is not None:
            det_bin = next_observation.last_detection_bin
            if self.last_predicted_bin is not None and det_bin != self.last_predicted_bin:
                instant_surprise = max(instant_surprise, 0.80)

        # Smooth update of surprise
        self.surprise = (
            self.surprise_decay * self.surprise + (1.0 - self.surprise_decay) * instant_surprise
        )

    def arbitrate(
        self,
        observation: SchedulerObservation,
        q_values: np.ndarray,
        ca_scores: np.ndarray,
        ca_details: dict[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        """Executes transparent arbitration and returns chosen bin index and full telemetry."""
        n = self.num_bins

        # Defensive dimension alignment
        if len(q_values) != n:
            if len(q_values) > n:
                q_values = q_values[:n]
            else:
                fill = float(np.min(q_values)) if len(q_values) > 0 else 0.0
                q_values = np.pad(q_values, (0, n - len(q_values)), constant_values=fill)
        if len(ca_scores) != n:
            if len(ca_scores) > n:
                ca_scores = ca_scores[:n]
            else:
                fill = float(np.min(ca_scores)) if len(ca_scores) > 0 else 0.0
                ca_scores = np.pad(ca_scores, (0, n - len(ca_scores)), constant_values=fill)

        # 1. DDQN Confidence Calculation
        # Combines z-score prominence of top Q-value above baseline with top-1 vs top-2 separation
        sorted_q_idx = np.argsort(q_values)[::-1]
        best_ddqn_bin = int(sorted_q_idx[0])
        second_ddqn_bin = int(sorted_q_idx[1]) if n > 1 else best_ddqn_bin
        
        q_mean = float(np.mean(q_values))
        q_std = float(np.std(q_values))
        if q_std > 1e-4:
            z_score = (float(q_values[best_ddqn_bin]) - q_mean) / q_std
            z_conf = float(np.clip((z_score - 1.2) / 2.0, 0.0, 1.0))
        else:
            z_conf = 0.0

        q_spread = float(q_values[best_ddqn_bin] - q_values[sorted_q_idx[-1]])
        if q_spread > 1e-4:
            gap_conf = float(np.clip((q_values[best_ddqn_bin] - q_values[second_ddqn_bin]) / q_spread, 0.0, 1.0))
        else:
            gap_conf = 0.0

        ddqn_confidence = float(max(z_conf, gap_conf))

        # 2. CA Confidence Calculation
        # Transition probability peak above uniform (1/N)
        trans_probs = ca_details.get("transition_probs", np.full(n, 1.0 / n, dtype=np.float32))
        best_ca_bin = int(np.argmax(ca_scores))
        max_trans = float(np.max(trans_probs))
        uniform_baseline = 1.0 / float(n)
        if max_trans > uniform_baseline:
            ca_trans_conf = (max_trans - uniform_baseline) / max(1.0 - uniform_baseline, 1e-4)
        else:
            ca_trans_conf = 0.0

        # Activity level peak
        act_scores = ca_details.get("activity_scores", np.zeros(n, dtype=np.float32))
        max_act = float(np.max(act_scores)) if len(act_scores) > 0 else 0.0

        ca_confidence = float(np.clip(max(ca_trans_conf, max_act * 0.8), 0.0, 1.0))

        # 3. Recent Detection Performance
        det_hist = observation.recent_detection_history or ()
        if det_hist:
            recent_det_rate = float(sum(1 for d in det_hist if bool(d))) / float(len(det_hist))
        else:
            recent_det_rate = 0.0

        # 4. Transition Consistency Estimation
        freq_hist = observation.recent_frequency_history or ()
        consistency = 0.0
        if len(freq_hist) >= 3:
            matches = sum(1 for i in range(1, len(freq_hist)) if freq_hist[i] == freq_hist[i - 1])
            consistency = max(consistency, float(matches) / float(len(freq_hist) - 1))
            if len(freq_hist) >= 4:
                cycle_matches = sum(1 for i in range(2, len(freq_hist)) if freq_hist[i] == freq_hist[i - 2])
                consistency = max(consistency, float(cycle_matches) / float(len(freq_hist) - 2))
        if best_ddqn_bin == best_ca_bin:
            consistency = max(consistency, 0.6)
        if recent_det_rate >= 0.20:
            consistency = max(consistency, 0.5)

        # 5. Determine Mode and Dynamic Weights
        # Effective surprise is moderated when policy is actively intercepting targets
        if recent_det_rate >= 0.15:
            effective_surprise = self.surprise * max(0.0, 1.0 - 1.5 * recent_det_rate)
        else:
            effective_surprise = self.surprise

        # Rule-based arbitration logic:
        # A. High DDQN confidence + good consistency + low surprise -> DDQN EXPLOIT
        if (
            ddqn_confidence >= self.ddqn_conf_threshold
            and effective_surprise <= self.surprise_threshold
            and (consistency >= self.consistency_threshold or recent_det_rate >= 0.15)
        ):
            mode = ArbitrationMode.DDQN_EXPLOIT
            w_ddqn = 0.85 + 0.15 * ddqn_confidence
            w_ca = 1.0 - w_ddqn

        # B. High surprise / novelty OR strong CA confidence with weak DDQN -> CA ADAPT
        elif (
            effective_surprise > self.surprise_threshold
            or (ca_confidence >= self.ca_conf_threshold and ddqn_confidence < self.ddqn_conf_threshold)
        ):
            mode = ArbitrationMode.CA_ADAPT
            w_ca = 0.80 + 0.20 * min(ca_confidence, max(effective_surprise, 0.1))
            w_ddqn = 1.0 - w_ca

        # C. Both confidences very low -> EXPLORE DISCOVERY
        elif ddqn_confidence < 0.10 and ca_confidence < 0.10 and recent_det_rate < 0.10:
            mode = ArbitrationMode.EXPLORE_DISCOVERY

        # D. Otherwise -> BLENDED
        else:
            mode = ArbitrationMode.BLENDED
            denom = ddqn_confidence + ca_confidence + 1e-4
            raw_w_ddqn = (ddqn_confidence / denom) * (1.0 - 0.5 * effective_surprise)
            w_ddqn = float(np.clip(raw_w_ddqn, 0.20, 0.80))
            w_ca = 1.0 - w_ddqn

        # 6. Normalize Scores and Select Final Action
        if mode == ArbitrationMode.DDQN_EXPLOIT:
            chosen_bin = best_ddqn_bin
            w_ddqn = 1.0
            w_ca = 0.0
            final_scores = np.zeros(n, dtype=np.float32)
            final_scores[chosen_bin] = 1.0

        elif mode == ArbitrationMode.CA_ADAPT:
            chosen_bin = best_ca_bin
            w_ddqn = 0.0
            w_ca = 1.0
            final_scores = np.zeros(n, dtype=np.float32)
            final_scores[chosen_bin] = 1.0

        elif mode == ArbitrationMode.EXPLORE_DISCOVERY:
            exp_scores = ca_details.get("exploration_scores", np.zeros(n, dtype=np.float32))
            chosen_bin = int(np.argmax(exp_scores))
            w_ca = 0.70
            w_ddqn = 0.30
            final_scores = exp_scores

        else:
            # BLENDED Mode: smooth normalized combination
            q_min, q_max = float(np.min(q_values)), float(np.max(q_values))
            if q_max - q_min > 1e-5:
                norm_q = (q_values - q_min) / (q_max - q_min)
            else:
                norm_q = np.full(n, 1.0 / n, dtype=np.float32)

            ca_min, ca_max = float(np.min(ca_scores)), float(np.max(ca_scores))
            if ca_max - ca_min > 1e-5:
                norm_ca = (ca_scores - ca_min) / (ca_max - ca_min)
            else:
                norm_ca = np.full(n, 1.0 / n, dtype=np.float32)

            final_scores = w_ddqn * norm_q + w_ca * norm_ca
            chosen_bin = int(np.argmax(final_scores))
        self.last_predicted_bin = best_ddqn_bin if w_ddqn > 0.5 else best_ca_bin
        self.last_chosen_branch = "DDQN" if w_ddqn > 0.5 else "CA"

        telemetry = {
            "timestamp": observation.timestamp,
            "selected_bin": chosen_bin,
            "CA_best_action": best_ca_bin,
            "DDQN_best_action": best_ddqn_bin,
            "CA_confidence": ca_confidence,
            "DDQN_confidence": ddqn_confidence,
            "novelty_surprise": float(self.surprise),
            "transition_consistency": float(consistency),
            "recent_detection_rate": float(recent_det_rate),
            "CA_weight": float(w_ca),
            "DDQN_weight": float(w_ddqn),
            "arbitration_mode": mode.value,
            "chosen_branch": self.last_chosen_branch,
            "final_score": float(final_scores[chosen_bin]),
        }
        self.last_decision_telemetry = telemetry
        return chosen_bin, telemetry
