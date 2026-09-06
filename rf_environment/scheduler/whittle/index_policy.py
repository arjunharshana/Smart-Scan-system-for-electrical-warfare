from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.scheduler.whittle.belief_state import BanditBeliefTracker
from rf_environment.scheduler.whittle.config import WhittleConfig
from rf_environment.scheduler.whittle.transition_estimator import ObservableTransitionEstimator


class WhittleIndexPolicy:
    """Computes the per-arm Whittle-style index heuristic and action selection.

    Ablation modes supported:
    - W0: Belief only
    - W1: Belief + Recency (transition likelihood)
    - W2: Belief + Recency + Uncertainty (exploration bonus)
    - W3: Full Model (Belief + Recency + Uncertainty + Dwell Persistence + Periodicity)
    """

    def __init__(
        self,
        config: WhittleConfig | None = None,
    ) -> None:
        self.config = config or WhittleConfig()

    def compute_indices(
        self,
        belief_tracker: BanditBeliefTracker,
        transition_estimator: ObservableTransitionEstimator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Computes index values for all N frequency arms and returns full diagnostic breakdown."""
        n = belief_tracker.num_bins
        mode = self.config.ablation_mode

        beliefs = belief_tracker.get_beliefs()
        uncertainties = belief_tracker.get_uncertainties()
        times_since_scan = belief_tracker.get_times_since_scan()
        times_since_det = belief_tracker.get_times_since_detection()

        # 1. Exploitation term (Belief)
        v_belief = self.config.w_belief * beliefs

        # 2. Recency / Empirical Transition term
        trans_probs = transition_estimator.get_transition_probabilities(
            from_bin=belief_tracker.last_detected_bin
        )
        v_recency = self.config.w_recency * trans_probs

        # 3. Uncertainty / Information Coverage term
        v_uncertainty = self.config.w_uncertainty * uncertainties

        # 4. Dwell Persistence term
        v_dwell = np.zeros(n, dtype=np.float32)
        last_det = belief_tracker.last_detected_bin
        if last_det is not None and 0 <= last_det < n:
            last_arm = belief_tracker.arms[last_det]
            # If currently in an active dwell run, compute remaining dwell fraction
            if last_arm.consecutive_detections > 0:
                est_dwell = max(last_arm.estimated_dwell, 1.0)
                rem_frac = max(0.0, 1.0 - (last_arm.consecutive_detections / est_dwell))
                v_dwell[last_det] = float(self.config.w_dwell * rem_frac)

        # 5. Periodicity / Return Interval Synchronization term
        v_periodic = np.zeros(n, dtype=np.float32)
        sigma2 = 2.0 * (self.config.sigma_periodic ** 2)
        for i, arm in enumerate(belief_tracker.arms):
            if arm.estimated_interval > 1.0 and arm.time_since_detection > 0:
                elapsed = float(arm.time_since_detection)
                # Distance to closest multiple of interval
                rem_dist = abs(elapsed - arm.estimated_interval)
                period_match = float(np.exp(- (rem_dist ** 2) / max(sigma2, 1e-4)))
                v_periodic[i] = float(self.config.w_periodic * period_match)

        # Assemble index based on ablation mode
        if mode == "W0":
            indices = v_belief.copy()
        elif mode == "W1":
            indices = v_belief + v_recency
        elif mode == "W2":
            indices = v_belief + v_recency + v_uncertainty
        elif mode == "W3":
            indices = v_belief + v_recency + v_uncertainty + v_dwell + v_periodic
        else:
            raise ValueError(f"Unknown Whittle ablation mode: {mode}")

        # Guard against NaN/Inf
        indices = np.nan_to_num(indices, nan=0.0, posinf=1.0, neginf=0.0)

        # Deterministic tie-breaking: prioritize higher index, then smaller bin index
        sorted_bins = np.lexsort((np.arange(n), -indices))
        selected_bin = int(sorted_bins[0])

        top_k = [int(b) for b in sorted_bins[:min(5, n)]]

        details = {
            "ablation_mode": mode,
            "selected_bin": selected_bin,
            "top_k_bins": top_k,
            "selected_index": float(indices[selected_bin]),
            "indices": indices.tolist(),
            "v_belief": v_belief.tolist(),
            "v_recency": v_recency.tolist(),
            "v_uncertainty": v_uncertainty.tolist(),
            "v_dwell": v_dwell.tolist(),
            "v_periodic": v_periodic.tolist(),
            "selected_components": {
                "belief": float(v_belief[selected_bin]),
                "recency": float(v_recency[selected_bin]),
                "uncertainty": float(v_uncertainty[selected_bin]),
                "dwell": float(v_dwell[selected_bin]),
                "periodic": float(v_periodic[selected_bin]),
                "total_index": float(indices[selected_bin]),
            },
        }

        return indices, details
