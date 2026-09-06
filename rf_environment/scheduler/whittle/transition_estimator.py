from __future__ import annotations

import numpy as np


class ObservableTransitionEstimator:
    """Estimates empirical channel-to-channel transition probabilities solely from receiver detections.

    Guarantees:
    - Zero ground truth access.
    - Only tracks transitions between successively confirmed detector hits.
    - Provides smoothed transition vectors for any source bin.
    """

    def __init__(self, num_bins: int, smoothing: float = 0.05) -> None:
        self.num_bins = int(num_bins)
        self.smoothing = float(smoothing)
        self.counts: np.ndarray = np.zeros((self.num_bins, self.num_bins), dtype=np.float32)
        self.last_hit_bin: int | None = None

    def reset(self) -> None:
        """Resets all empirical transition counts with zero cross-episode leakage."""
        self.counts.fill(0.0)
        self.last_hit_bin = None

    def record_detection(self, current_hit_bin: int) -> None:
        """Records an observed transition from the previous detected bin to the current detected bin."""
        if not 0 <= current_hit_bin < self.num_bins:
            return

        if self.last_hit_bin is not None and 0 <= self.last_hit_bin < self.num_bins:
            # We record transition if it changed bins or if successive detection confirmed channel
            self.counts[self.last_hit_bin, current_hit_bin] += 1.0

        self.last_hit_bin = current_hit_bin

    def get_transition_probabilities(self, from_bin: int | None = None) -> np.ndarray:
        """Returns the empirical next-bin probability distribution P(next_bin | from_bin)."""
        src = from_bin if from_bin is not None else self.last_hit_bin

        if src is None or not 0 <= src < self.num_bins:
            # Uniform prior
            return np.full(self.num_bins, 1.0 / self.num_bins, dtype=np.float32)

        row = self.counts[src]
        total = float(np.sum(row))
        denom = total + self.smoothing * self.num_bins
        probs = (row + self.smoothing) / max(denom, 1e-8)
        return probs.astype(np.float32)
