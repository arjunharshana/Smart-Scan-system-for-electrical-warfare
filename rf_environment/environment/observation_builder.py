from __future__ import annotations

from collections import deque
from typing import Sequence

from rf_environment.domain.state import SchedulerObservation


class ObservationBuilder:
    """Constructs the canonical SchedulerObservation.

    Strict Invariants:
    1. Only contains legitimate receiver/detector observables up to decision time.
    2. Zero ground truth (no true emitter frequencies, trajectories, IDs, or opportunity labels).
    3. Zero future lookahead.
    4. Immutable frozen dataclass output.
    """

    def __init__(self, num_bins: int, history_length: int = 10) -> None:
        self.num_bins = int(num_bins)
        self.history_length = max(1, int(history_length))

        self.recent_detections: deque[bool] = deque(maxlen=self.history_length)
        self.recent_frequencies: deque[int] = deque(maxlen=self.history_length)

        self.scan_count_by_bin: list[int] = [0] * self.num_bins
        self.last_scanned_time_by_bin: list[float | None] = [None] * self.num_bins

        self.last_detection: bool = False
        self.last_detection_bin: int | None = None
        self.last_detection_strength: float | None = None
        self.last_detection_time: float | None = None

    def reset(self) -> None:
        self.recent_detections.clear()
        self.recent_frequencies.clear()
        self.scan_count_by_bin = [0] * self.num_bins
        self.last_scanned_time_by_bin = [None] * self.num_bins
        self.last_detection = False
        self.last_detection_bin = None
        self.last_detection_strength = None
        self.last_detection_time = None

    def build_initial_observation(
        self,
        current_frequency_bin: int = 0,
    ) -> SchedulerObservation:
        """Constructs an initial observation at simulation start (t=0)."""
        return SchedulerObservation(
            timestamp=0.0,
            current_frequency_bin=current_frequency_bin,
            last_detection=False,
            last_detection_bin=None,
            last_detection_strength=None,
            recent_detection_history=(),
            recent_frequency_history=(),
            scan_count_by_bin=tuple([0] * self.num_bins),
            time_since_scan_by_bin=tuple([999.0] * self.num_bins),
            time_since_last_detection=999.0,
        )

    def step(
        self,
        timestamp: float,
        scanned_bin: int,
        detected: bool,
        signal_strength: float | None = None,
    ) -> SchedulerObservation:
        """Updates receiver history and returns the canonical immutable SchedulerObservation."""
        # Update temporal history
        self.recent_detections.append(detected)
        self.recent_frequencies.append(scanned_bin)

        # Update per-bin scan stats
        if 0 <= scanned_bin < self.num_bins:
            self.scan_count_by_bin[scanned_bin] += 1
            self.last_scanned_time_by_bin[scanned_bin] = timestamp

        # Update detection state
        self.last_detection = detected
        if detected:
            self.last_detection_bin = scanned_bin
            self.last_detection_strength = signal_strength
            self.last_detection_time = timestamp

        # Derived timing
        time_since_last_det = (
            float(timestamp - self.last_detection_time)
            if self.last_detection_time is not None
            else 999.0
        )

        time_since_scan = tuple(
            float(timestamp - t) if t is not None else 999.0
            for t in self.last_scanned_time_by_bin
        )

        return SchedulerObservation(
            timestamp=timestamp,
            current_frequency_bin=scanned_bin,
            last_detection=detected,
            last_detection_bin=self.last_detection_bin,
            last_detection_strength=self.last_detection_strength,
            recent_detection_history=tuple(self.recent_detections),
            recent_frequency_history=tuple(self.recent_frequencies),
            scan_count_by_bin=tuple(self.scan_count_by_bin),
            time_since_scan_by_bin=time_since_scan,
            time_since_last_detection=time_since_last_det,
        )
