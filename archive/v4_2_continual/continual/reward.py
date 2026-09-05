from __future__ import annotations

import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation


class ObservableOnlineReward:
    """Observable-only online reward calculator for V4.2 continual learning.

    Invariants:
    1. Consumes strictly detector/receiver observables from SchedulerObservation and ScanAction.
    2. Zero access to emitter state, true frequency, transmission flags, or opportunity labels.
    3. Fully deterministic given the same sequence of observations and actions.

    Formulation:
        r_total = r_det + r_strength + r_persist + r_step + r_unprod

    Where:
        r_det = +1.00 if last_detection is True else 0.00
        r_strength = +0.15 * clip((P_rx + 90.0) / 40.0, 0.0, 1.0) if last_detection else 0.00
        r_persist = +0.10 if last_detection is True and consecutive detection on same bin
        r_step = -0.05 if last_detection is False else 0.00
        r_unprod = -0.05 * min(consecutive_misses_on_bin, 3) if last_detection is False
    """

    def __init__(self, num_bins: int) -> None:
        self.num_bins = int(num_bins)
        self.consecutive_misses_by_bin = [0] * self.num_bins
        self.last_scanned_bin: int | None = None
        self.last_detection_bin: int | None = None

    def compute_reward(
        self,
        observation: SchedulerObservation,
        action: ScanAction | int,
    ) -> float:
        scanned_bin = action.frequency_bin if isinstance(action, ScanAction) else int(action)
        scanned_bin = min(max(scanned_bin, 0), self.num_bins - 1)

        detected = bool(observation.last_detection)
        rx_power = observation.last_detection_strength

        # 1. Detection Benefit
        if detected:
            r_det = 1.00
            # Strength shaping (same calibrated receiver scale as R4)
            if rx_power is not None:
                q = float(np.clip((rx_power + 90.0) / 40.0, 0.0, 1.0))
                r_strength = 0.15 * q
            else:
                r_strength = 0.00

            # Persistence bonus: consecutive detection on the same frequency bin
            if self.last_detection_bin == scanned_bin:
                r_persist = 0.10
            else:
                r_persist = 0.00

            r_step = 0.00
            r_unprod = 0.00

            # Reset misses on successful scan
            self.consecutive_misses_by_bin[scanned_bin] = 0
            self.last_detection_bin = scanned_bin

        else:
            r_det = 0.00
            r_strength = 0.00
            r_persist = 0.00
            r_step = -0.05

            # Increment unproductive scan streak for this bin
            miss_streak = self.consecutive_misses_by_bin[scanned_bin]
            r_unprod = -0.05 * float(min(miss_streak, 3))
            self.consecutive_misses_by_bin[scanned_bin] += 1
            self.last_detection_bin = None

        self.last_scanned_bin = scanned_bin

        total_reward = r_det + r_strength + r_persist + r_step + r_unprod
        return float(total_reward)

    def reset(self) -> None:
        """Resets online reward tracking state."""
        self.consecutive_misses_by_bin = [0] * self.num_bins
        self.last_scanned_bin = None
        self.last_detection_bin = None
