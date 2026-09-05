from __future__ import annotations

import math
import numpy as np

from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.rl.encoder import ObservationEncoder


class TemporalObservationEncoder(ObservationEncoder):
    """Extended Observation Encoder for V3.1 DDQN Robustness Study.

    Derives strictly legitimate receiver-observable temporal features from
    canonical SchedulerObservation history without accessing any simulator
    ground truth or future state.

    Configurable options:
    - include_absolute_time: If False, omits normalized episode timestamp (min(t/1000, 1)).
    - include_freq_movement: frequency delta [-1, 1] and frequency direction {-1, 0, 1}.
    - include_det_rate: rolling detection rate in recent history [0, 1].
    - include_persistence: consecutive detection count on current bin / H [0, 1].
    - include_time_between_detections: steps between recent detections / H [0, 1].
    - include_consistency: transition repeatability and unique bin count [0, 1].
    - include_periodic_features: generic multi-scale Fourier encoding sin/cos(2*pi*t / P_k).
    """

    GENERIC_PERIODS = (4.0, 8.0, 16.0, 32.0, 64.0)

    def __init__(
        self,
        num_bins: int,
        history_length: int = 10,
        include_absolute_time: bool = True,
        include_freq_movement: bool = False,
        include_det_rate: bool = False,
        include_persistence: bool = False,
        include_time_between_detections: bool = False,
        include_consistency: bool = False,
        include_periodic_features: bool = False,
    ) -> None:
        super().__init__(num_bins=num_bins, history_length=history_length)
        self.include_absolute_time = bool(include_absolute_time)
        self.include_freq_movement = bool(include_freq_movement)
        self.include_det_rate = bool(include_det_rate)
        self.include_persistence = bool(include_persistence)
        self.include_time_between_detections = bool(include_time_between_detections)
        self.include_consistency = bool(include_consistency)
        self.include_periodic_features = bool(include_periodic_features)

        # Base dimension is (26 + 4*N) if include_absolute_time is True, else (25 + 4*N)
        base_dim = (26 if self.include_absolute_time else 25) + 4 * self.num_bins

        extra_dim = 0
        if self.include_freq_movement:
            extra_dim += 2
        if self.include_det_rate:
            extra_dim += 1
        if self.include_persistence:
            extra_dim += 1
        if self.include_time_between_detections:
            extra_dim += 1
        if self.include_consistency:
            extra_dim += 2
        if self.include_periodic_features:
            extra_dim += 2 * len(self.GENERIC_PERIODS)

        self.feature_dim = base_dim + extra_dim

    @classmethod
    def create_encoder_a(cls, num_bins: int, history_length: int = 10) -> TemporalObservationEncoder:
        """Encoder A: Exact V3.0 baseline observation encoder."""
        return cls(num_bins=num_bins, history_length=history_length, include_absolute_time=True)

    @classmethod
    def create_encoder_b(cls, num_bins: int, history_length: int = 10) -> TemporalObservationEncoder:
        """Encoder B: Encoder A + all legitimate receiver-derived temporal features."""
        return cls(
            num_bins=num_bins,
            history_length=history_length,
            include_absolute_time=True,
            include_freq_movement=True,
            include_det_rate=True,
            include_persistence=True,
            include_time_between_detections=True,
            include_consistency=True,
        )

    @classmethod
    def create_encoder_c(cls, num_bins: int, history_length: int = 10) -> TemporalObservationEncoder:
        """Encoder C: No absolute timestamp, retaining relative timing and temporal features."""
        return cls(
            num_bins=num_bins,
            history_length=history_length,
            include_absolute_time=False,
            include_freq_movement=True,
            include_det_rate=True,
            include_persistence=True,
            include_time_between_detections=True,
            include_consistency=True,
        )

    @classmethod
    def create_encoder_d(cls, num_bins: int, history_length: int = 10) -> TemporalObservationEncoder:
        """Encoder D: Generic multi-scale Fourier harmonic periodic encoding."""
        return cls(
            num_bins=num_bins,
            history_length=history_length,
            include_absolute_time=False,
            include_periodic_features=True,
        )

    def encode(self, observation: SchedulerObservation) -> np.ndarray:
        """Encodes SchedulerObservation into a 1D np.float32 vector."""
        vec = np.zeros(self.feature_dim, dtype=np.float32)
        idx = 0

        # 1. Normalized timestamp [1] (optional)
        t = float(observation.timestamp) if observation.timestamp is not None else 0.0
        if self.include_absolute_time:
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
            norm_strength = (float(strength) + 90.0) / 40.0
            vec[idx] = min(max(norm_strength, 0.0), 1.0)
        else:
            vec[idx] = 0.0
        idx += 1

        vec[idx] = 1.0 if strength is not None else 0.0
        idx += 1

        # 6. Recent detection history [H]
        det_hist = observation.recent_detection_history or ()
        h_len = min(len(det_hist), self.history_length)
        if h_len > 0:
            tail = det_hist[-h_len:]
            offset = self.history_length - h_len
            for i, val in enumerate(tail):
                vec[idx + offset + i] = 1.0 if bool(val) else 0.0
        idx += self.history_length

        # 7. Recent frequency history [H]
        freq_hist = observation.recent_frequency_history or ()
        f_len = min(len(freq_hist), self.history_length)
        vec[idx : idx + self.history_length] = -1.0
        max_bin_denom = max(self.num_bins - 1, 1)
        if f_len > 0:
            tail_f = freq_hist[-f_len:]
            offset = self.history_length - f_len
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

        # --- DERIVED TEMPORAL FEATURES (strictly receiver-observable) ---

        # A. Frequency Movement (delta and direction) [2]
        if self.include_freq_movement:
            delta = 0.0
            direction = 0.0
            if len(freq_hist) >= 2:
                b_curr = freq_hist[-1]
                b_prev = freq_hist[-2]
                if b_curr is not None and b_prev is not None:
                    raw_diff = float(b_curr - b_prev)
                    delta = raw_diff / float(max_bin_denom)
                    direction = float(np.sign(raw_diff))
            vec[idx] = float(np.clip(delta, -1.0, 1.0))
            vec[idx + 1] = direction
            idx += 2

        # B. Detection Rate in recent history [1]
        if self.include_det_rate:
            if det_hist:
                rate = float(sum(1.0 for d in det_hist if bool(d))) / float(len(det_hist))
            else:
                rate = 0.0
            vec[idx] = float(np.clip(rate, 0.0, 1.0))
            idx += 1

        # C. Detection Persistence (consecutive detections on current bin) [1]
        if self.include_persistence:
            consecutive = 0
            for d in reversed(det_hist):
                if bool(d):
                    consecutive += 1
                else:
                    break
            vec[idx] = min(float(consecutive) / float(self.history_length), 1.0)
            idx += 1

        # D. Time Between Recent Detections [1]
        if self.include_time_between_detections:
            det_indices = [i for i, d in enumerate(det_hist) if bool(d)]
            if len(det_indices) >= 2:
                interval = float(det_indices[-1] - det_indices[-2])
                norm_interval = min(interval / float(self.history_length), 1.0)
            else:
                norm_interval = 1.0  # long / no recent interval
            vec[idx] = norm_interval
            idx += 1

        # E. Transition Consistency & Scan Dispersion [2]
        if self.include_consistency:
            consistency = 0.0
            if len(freq_hist) >= 3:
                diff1 = freq_hist[-1] - freq_hist[-2]
                diff2 = freq_hist[-2] - freq_hist[-3]
                consistency = 1.0 if diff1 == diff2 else 0.0
            vec[idx] = consistency
            idx += 1

            unique_count = len({b for b in freq_hist if b is not None})
            max_possible = min(len(freq_hist), self.num_bins) if freq_hist else 1
            dispersion = float(unique_count) / float(max(max_possible, 1))
            vec[idx] = float(np.clip(dispersion, 0.0, 1.0))
            idx += 1

        # F. Generic Periodic Fourier Encoding [2 * len(GENERIC_PERIODS)]
        if self.include_periodic_features:
            for p in self.GENERIC_PERIODS:
                phase = 2.0 * math.pi * (t % p) / p
                vec[idx] = float(math.sin(phase))
                vec[idx + 1] = float(math.cos(phase))
                idx += 2

        assert idx == self.feature_dim, f"Encoder dim mismatch: {idx} != {self.feature_dim}"
        return vec
