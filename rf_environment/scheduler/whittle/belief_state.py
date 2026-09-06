from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from rf_environment.scheduler.whittle.config import WhittleConfig


@dataclass
class ArmState:
    """Belief state and observable telemetry for a single frequency arm."""

    bin_idx: int
    belief: float
    uncertainty: float
    time_since_scan: int = 999
    time_since_detection: int = 999
    consecutive_detections: int = 0
    total_scans: int = 0
    total_detections: int = 0
    last_detection_time: float | None = None
    recent_intervals: list[float] = field(default_factory=list)
    estimated_dwell: float = 3.0
    estimated_interval: float = 0.0


class BanditBeliefTracker:
    """Maintains receiver-observable POMDP belief states and empirical timers for all frequency arms.

    Strict Invariants:
    1. Zero ground truth or future information.
    2. Only updates using confirmed detector observables (scan bin, detection bool, timestamp).
    3. Deterministic resets with zero cross-episode state retention.
    """

    def __init__(self, num_bins: int, config: WhittleConfig | None = None) -> None:
        self.num_bins = int(num_bins)
        self.config = config or WhittleConfig()

        if self.config.background_prior is not None:
            self.prior = float(self.config.background_prior)
        else:
            self.prior = 1.0 / max(self.num_bins, 1)
        self.arms: list[ArmState] = []
        self.t_step: int = 0
        self.last_scanned_bin: int | None = None
        self.last_detected_bin: int | None = None

        self.reset()

    def reset(self) -> None:
        """Resets all arm states to clean initial priors with zero leakage."""
        self.arms = [
            ArmState(
                bin_idx=i,
                belief=self.prior,
                uncertainty=1.0,
                time_since_scan=999,
                time_since_detection=999,
                consecutive_detections=0,
                total_scans=0,
                total_detections=0,
                last_detection_time=None,
                recent_intervals=[],
                estimated_dwell=float(self.config.default_dwell),
                estimated_interval=0.0,
            )
            for i in range(self.num_bins)
        ]
        self.t_step = 0
        self.last_scanned_bin = None
        self.last_detected_bin = None

    def get_beliefs(self) -> np.ndarray:
        return np.array([a.belief for a in self.arms], dtype=np.float32)

    def get_uncertainties(self) -> np.ndarray:
        return np.array([a.uncertainty for a in self.arms], dtype=np.float32)

    def get_times_since_scan(self) -> np.ndarray:
        return np.array([a.time_since_scan for a in self.arms], dtype=np.int32)

    def get_times_since_detection(self) -> np.ndarray:
        return np.array([a.time_since_detection for a in self.arms], dtype=np.int32)

    def update_observation(
        self,
        scanned_bin: int,
        detection: bool,
        timestamp: float | None = None,
    ) -> None:
        """Updates active and passive arms following an action and detector observation."""
        if not 0 <= scanned_bin < self.num_bins:
            return

        self.t_step += 1
        self.last_scanned_bin = scanned_bin
        p_d = self.config.p_detection
        p_fa = self.config.p_false_alarm
        t_now = timestamp if timestamp is not None else float(self.t_step)

        # 1. Update active scanned arm
        arm = self.arms[scanned_bin]
        arm.total_scans += 1
        arm.time_since_scan = 0
        arm.uncertainty = 0.0

        if detection:
            self.last_detected_bin = scanned_bin
            arm.total_detections += 1
            arm.consecutive_detections += 1

            # Bayesian update on hit: P(active | det)
            prior = arm.belief
            num = p_d * prior
            denom = num + p_fa * (1.0 - prior)
            arm.belief = float(np.clip(num / max(denom, 1e-8), 0.01, 0.99))

            # Inter-hit interval tracking
            if arm.last_detection_time is not None and t_now > arm.last_detection_time:
                interval = t_now - arm.last_detection_time
                # Only record interval if it represents a return from another hop (dt > 1)
                if arm.time_since_detection > 1:
                    arm.recent_intervals.append(interval)
                    if len(arm.recent_intervals) > 10:
                        arm.recent_intervals.pop(0)
                    # EWMA interval estimate
                    if arm.estimated_interval <= 0.0:
                        arm.estimated_interval = float(interval)
                    else:
                        arm.estimated_interval = 0.7 * arm.estimated_interval + 0.3 * interval

            arm.last_detection_time = t_now
            arm.time_since_detection = 0
        else:
            # Dwell tracking: if a detection streak just ended, update estimated dwell
            if arm.consecutive_detections > 0:
                streak = float(arm.consecutive_detections)
                arm.estimated_dwell = 0.7 * arm.estimated_dwell + 0.3 * streak
            arm.consecutive_detections = 0

            # Bayesian update on miss: P(active | no_det)
            prior = arm.belief
            num = (1.0 - p_d) * prior
            denom = num + (1.0 - p_fa) * (1.0 - prior)
            arm.belief = float(np.clip(num / max(denom, 1e-8), 0.001, 0.50))
            arm.time_since_detection += 1

        # 2. Update passive unscanned arms
        decay = self.config.belief_decay
        stale = max(self.config.stale_threshold, 1)

        for j in range(self.num_bins):
            if j == scanned_bin:
                continue
            p_arm = self.arms[j]
            p_arm.time_since_scan += 1
            p_arm.time_since_detection += 1

            # Passive belief decay toward background prior
            p_arm.belief = float(np.clip(p_arm.belief * decay + (1.0 - decay) * self.prior, 0.0, 1.0))

            # Uncertainty grows with unobserved duration
            p_arm.uncertainty = float(np.clip(p_arm.time_since_scan / stale, 0.0, 1.0))
