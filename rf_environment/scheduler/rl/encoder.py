from __future__ import annotations

import numpy as np

from rf_environment.domain.state import SchedulerObservation


class ObservationEncoder:
    """Canonical Observation Encoder for Reinforcement Learning Schedulers.

    Maps a canonical SchedulerObservation (10 fields) into a deterministic 1D
    np.float32 feature vector for neural network consumption.

    Architectural Invariants:
    1. Consumes strictly legitimate scheduler observables from SchedulerObservation.
    2. Zero ground truth, zero future lookahead, zero evaluator dependencies.
    3. Completely deterministic: identical observation -> identical float32 vector.
    4. Deterministic None handling: None values produce zeroed indicators, never NaN or Inf.
    5. Normalization: all features are scaled to [0.0, 1.0] or [-1.0, 1.0].

    Feature Layout (Dimension D = 26 + 4*N, where N = num_bins and H = 10):
    -----------------------------------------------------------------------------
    Index Range                 Dim     Description
    -----------------------------------------------------------------------------
    [0]                         1       Normalized timestamp: min(t / 1000.0, 1.0)
    [1 : 1 + N]                 N       Current frequency bin (one-hot of length N)
    [1 + N]                     1       Last detection flag (1.0 if True else 0.0)
    [2 + N : 2 + 2N]            N       Last detection bin (one-hot of length N, or 0s)
    [2 + 2N]                    1       Has last detection bin indicator (1.0 if not None)
    [3 + 2N]                    1       Last detection strength: clip((dBm + 90)/40, 0, 1)
    [4 + 2N]                    1       Has last detection strength indicator (1.0 if not None)
    [5 + 2N : 15 + 2N]          10      Recent detection history (padded H=10, 1.0/0.0)
    [15 + 2N : 25 + 2N]         10      Recent frequency history (padded H=10, b/(N-1) or -1.0)
    [25 + 2N : 25 + 3N]         N       Scan count by bin distribution (sums to 1.0 or 0s)
    [25 + 3N : 25 + 4N]         N       Time since scan by bin: min(t / 100.0, 1.0)
    [25 + 4N]                   1       Time since last detection: min(t / 100.0, 1.0)
    -----------------------------------------------------------------------------
    Total Dimension: 1 + N + 1 + N + 1 + 1 + 1 + 10 + 10 + N + N + 1 = 26 + 4*N
    """

    def __init__(self, num_bins: int, history_length: int = 10) -> None:
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        self.num_bins = int(num_bins)
        self.history_length = max(1, int(history_length))
        self.feature_dim = 26 + 4 * self.num_bins

    def encode(self, observation: SchedulerObservation) -> np.ndarray:
        """Encodes SchedulerObservation into a 1D np.float32 vector."""
        vec = np.zeros(self.feature_dim, dtype=np.float32)
        idx = 0

        # 1. Normalized timestamp [1]
        t = float(observation.timestamp) if observation.timestamp is not None else 0.0
        vec[idx] = min(max(t / 1000.0, 0.0), 1.0)
        idx += 1

        # 2. Current frequency bin one-hot [N]
        curr_bin = observation.current_frequency_bin
        if curr_bin is not None and 0 <= curr_bin < self.num_bins:
            vec[idx + curr_bin] = 1.0
        idx += self.num_bins

        # 3. Last detection flag [1]
        vec[idx] = 1.0 if bool(observation.last_detection) else 0.0
        idx += 1

        # 4. Last detection bin one-hot [N] and indicator [1]
        det_bin = observation.last_detection_bin
        if det_bin is not None and 0 <= det_bin < self.num_bins:
            vec[idx + det_bin] = 1.0
        idx += self.num_bins

        vec[idx] = 1.0 if det_bin is not None else 0.0
        idx += 1

        # 5. Last detection strength normalized [1] and indicator [1]
        strength = observation.last_detection_strength
        if strength is not None:
            # -90 dBm -> 0.0, -50 dBm -> 1.0
            norm_strength = (float(strength) + 90.0) / 40.0
            vec[idx] = min(max(norm_strength, 0.0), 1.0)
        else:
            vec[idx] = 0.0
        idx += 1

        vec[idx] = 1.0 if strength is not None else 0.0
        idx += 1

        # 6. Recent detection history [H=10]
        det_hist = observation.recent_detection_history or ()
        h_len = min(len(det_hist), self.history_length)
        # Pad with 0.0 on left, place most recent at right
        if h_len > 0:
            tail = det_hist[-h_len:]
            offset = self.history_length - h_len
            for i, val in enumerate(tail):
                vec[idx + offset + i] = 1.0 if bool(val) else 0.0
        idx += self.history_length

        # 7. Recent frequency history [H=10]
        freq_hist = observation.recent_frequency_history or ()
        f_len = min(len(freq_hist), self.history_length)
        # Padded elements initialized to -1.0
        vec[idx : idx + self.history_length] = -1.0
        if f_len > 0:
            tail_f = freq_hist[-f_len:]
            offset = self.history_length - f_len
            max_bin_denom = max(self.num_bins - 1, 1)
            for i, b in enumerate(tail_f):
                if b is not None and 0 <= b < self.num_bins:
                    vec[idx + offset + i] = float(b) / float(max_bin_denom)
        idx += self.history_length

        # 8. Scan count by bin distribution [N]
        scan_counts = observation.scan_count_by_bin or ()
        if scan_counts:
            raw_counts = np.array(scan_counts[: self.num_bins], dtype=np.float32)
            if len(raw_counts) < self.num_bins:
                padded = np.zeros(self.num_bins, dtype=np.float32)
                padded[: len(raw_counts)] = raw_counts
                raw_counts = padded
            total_scans = float(np.sum(raw_counts))
            if total_scans > 0.0:
                vec[idx : idx + self.num_bins] = raw_counts / total_scans
        idx += self.num_bins

        # 9. Time since scan by bin [N]
        time_since_scans = observation.time_since_scan_by_bin or ()
        if time_since_scans:
            raw_times = np.array(time_since_scans[: self.num_bins], dtype=np.float32)
            if len(raw_times) < self.num_bins:
                padded = np.full(self.num_bins, 100.0, dtype=np.float32)
                padded[: len(raw_times)] = raw_times
                raw_times = padded
            # Scale 0 -> 0.0, >=100.0 (or 999.0) -> 1.0
            vec[idx : idx + self.num_bins] = np.clip(raw_times / 100.0, 0.0, 1.0)
        else:
            vec[idx : idx + self.num_bins] = 1.0
        idx += self.num_bins

        # 10. Time since last detection [1]
        time_last_det = (
            float(observation.time_since_last_detection)
            if observation.time_since_last_detection is not None
            else 999.0
        )
        vec[idx] = min(max(time_last_det / 100.0, 0.0), 1.0)
        idx += 1

        assert idx == self.feature_dim, f"Encoder dimension mismatch: {idx} != {self.feature_dim}"
        return vec
