from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerObservation:
    """Canonical scheduler-facing observation schema.

    Represents ONLY what the receiver/detector has observed up to decision time.
    Strictly contains ZERO ground truth and ZERO future information.
    """

    timestamp: float
    current_frequency_bin: int

    last_detection: bool
    last_detection_bin: int | None
    last_detection_strength: float | None

    recent_detection_history: tuple[bool, ...]
    recent_frequency_history: tuple[int, ...]

    scan_count_by_bin: tuple[int, ...]
    time_since_scan_by_bin: tuple[float, ...]

    time_since_last_detection: float
