"""rf_environment/scheduler/belief/transition.py
Transition kernel and online transition estimator for V5.0 Bayesian Belief Scheduler.
"""

from __future__ import annotations

import numpy as np

from rf_environment.scheduler.belief.config import BeliefSchedulerConfig


class ObservableTransitionModel:
    """Manages the inter-frequency hopping transition matrix T(i -> j).

    Modes:
    - 'B0' / 'B1': Fixed uniform transition matrix across different channels:
      T[i, j] = 1 / (N - 1) for j != i, and 0 for j == i.
    - 'B2': Online empirical transition learning from detector confirmations:
      T[i, j] estimated via Dirichlet-smoothed transition counts C[i, j].
    """

    def __init__(
        self,
        num_bins: int,
        config: BeliefSchedulerConfig | None = None,
    ) -> None:
        self.num_bins = int(num_bins)
        self.config = config or BeliefSchedulerConfig()
        self.smoothing = float(self.config.transition_smoothing)
        self.mode = self.config.ablation_mode

        # Empirical transition counts C[i, j]
        self.counts = np.zeros((self.num_bins, self.num_bins), dtype=np.float64)
        self._last_detected_bin: int | None = None
        self._last_detected_time: float | None = None

        # Precomputed uniform transition matrix
        self._uniform_matrix = np.full(
            (self.num_bins, self.num_bins),
            1.0 / float(max(1, self.num_bins - 1)),
            dtype=np.float64,
        )
        np.fill_diagonal(self._uniform_matrix, 0.0)

    def reset(self) -> None:
        """Resets all transition counts with zero cross-episode leakage."""
        self.counts.fill(0.0)
        self._last_detected_bin = None
        self._last_detected_time = None

    def record_detection(self, frequency_bin: int, timestamp: float) -> None:
        """Records a confirmed receiver detection on frequency_bin.
        Updates transition counts if this detection represents a hop from a previous detection.
        """
        if self._last_detected_bin is not None:
            prev_bin = self._last_detected_bin
            # Only record transition if bin changed or reasonable time elapsed
            if prev_bin != frequency_bin:
                self.counts[prev_bin, frequency_bin] += 1.0

        self._last_detected_bin = frequency_bin
        self._last_detected_time = timestamp

    def get_transition_matrix(self) -> np.ndarray:
        """Returns the N x N transition probability matrix T where T[i, j] = P(f_{t+1} = j | f_t = i)."""
        if self.mode in {"B0", "B1"}:
            # Fixed uniform transitions
            return self._uniform_matrix.copy()

        # B2: Empirical transition matrix with Dirichlet prior
        alpha = self.smoothing
        # Prior: uniform over off-diagonal elements
        prior = np.full((self.num_bins, self.num_bins), alpha, dtype=np.float64)
        np.fill_diagonal(prior, 0.0)

        effective_counts = self.counts + prior
        row_sums = np.sum(effective_counts, axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums <= 0.0] = 1.0

        t_matrix = effective_counts / row_sums
        return t_matrix
