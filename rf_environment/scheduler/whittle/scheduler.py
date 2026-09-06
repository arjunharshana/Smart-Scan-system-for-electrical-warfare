from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler
from rf_environment.scheduler.whittle.belief_state import BanditBeliefTracker
from rf_environment.scheduler.whittle.config import WhittleConfig
from rf_environment.scheduler.whittle.index_policy import WhittleIndexPolicy
from rf_environment.scheduler.whittle.transition_estimator import ObservableTransitionEstimator


class WhittleScheduler(BaseScheduler):
    """Whittle-Style Heuristic Index Scheduler for Partially Observable RF Scanning.

    Treats each frequency bin as a restless bandit arm under partial observability.
    Calculates dynamic indices balancing:
    1. Belief (Exploitation)
    2. Recency / Empirical Transition likelihood
    3. Uncertainty / Information coverage
    4. Dwell duration persistence
    5. Periodicity / Return interval synchronization

    Architectural Invariants:
    1. Zero ground-truth leakage: Operates strictly on confirmed detector observables.
    2. Deterministic tie-breaking: Always resolves index ties to the lowest frequency bin index.
    3. $O(N)$ decision complexity: Computes all arm indices with minimal latency.
    4. Fully interpretable: Decomposes every scanning decision into its mathematical components.
    """

    name = "whittle_style"
    category = "bandit"

    def __init__(
        self,
        bands_hz: list[float],
        config: WhittleConfig | dict[str, Any] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(bands_hz)
        self.num_bins = len(self.bands_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if isinstance(config, WhittleConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = WhittleConfig.from_dict(config)
        else:
            cfg_dict = dict(kwargs)
            self.config = WhittleConfig.from_dict(cfg_dict)

        self.belief_tracker = BanditBeliefTracker(num_bins=self.num_bins, config=self.config)
        self.transition_estimator = ObservableTransitionEstimator(
            num_bins=self.num_bins, smoothing=self.config.transition_smoothing
        )
        self.policy = WhittleIndexPolicy(config=self.config)

        self._last_processed_timestamp: float | None = None
        self._last_selected_bin: int = 0

    def reset(self) -> None:
        """Resets all arm beliefs, timers, transition tables, and telemetry with zero cross-episode leakage."""
        super().reset()
        self.belief_tracker.reset()
        self.transition_estimator.reset()
        self._last_processed_timestamp = None
        self._last_selected_bin = 0

    def select_bin(self, observation: SchedulerObservation | Any = None) -> int:
        """Calculates Whittle-style indices across all N arms and selects the maximum index bin."""
        if isinstance(observation, SchedulerObservation):
            # Sync belief tracker if observation is newer than last processed transition
            if (
                self._last_processed_timestamp is None
                or observation.timestamp > self._last_processed_timestamp
            ):
                if self._last_processed_timestamp is not None:
                    # Previous scan produced observation.last_detection
                    prev_bin = observation.current_frequency_bin
                    prev_det = bool(observation.last_detection)
                    self.belief_tracker.update_observation(
                        scanned_bin=prev_bin,
                        detection=prev_det,
                        timestamp=observation.timestamp,
                    )
                    if prev_det:
                        self.transition_estimator.record_detection(prev_bin)

                self._last_processed_timestamp = observation.timestamp

        # Compute multi-term indices
        indices, details = self.policy.compute_indices(
            belief_tracker=self.belief_tracker,
            transition_estimator=self.transition_estimator,
        )

        chosen_bin = details["selected_bin"]
        self._last_selected_bin = chosen_bin
        self.last_selected_bin = chosen_bin
        self.last_indices = indices
        self.last_selected = self.bands_hz[chosen_bin]


        # Log detailed interpretable explanation
        comp = details["selected_components"]
        reason = (
            f"Whittle-Style Index={comp['total_index']:.3f} "
            f"[Belief={comp['belief']:.2f}, Recency={comp['recency']:.2f}, "
            f"Uncertainty={comp['uncertainty']:.2f}, Dwell={comp['dwell']:.2f}, "
            f"Periodic={comp['periodic']:.2f}]"
        )

        self.last_explanation = {
            "rule": "whittle_style",
            "ablation_mode": details["ablation_mode"],
            "selected_bin": chosen_bin,
            "action_mhz": self.last_selected / 1e6,
            "selected_index": details["selected_index"],
            "components": comp,
            "top_k_bins": details["top_k_bins"],
            "indices": details["indices"],
            "reason": reason,
        }

        return chosen_bin

    def update_policy(
        self,
        observation: SchedulerObservation,
        action: ScanAction | int,
        reward: float,
        next_observation: SchedulerObservation,
        done: bool,
    ) -> None:
        """Processes transition outcome and updates active arm belief and passive arm aging."""
        action_idx = action.frequency_bin if isinstance(action, ScanAction) else int(action)
        detection = bool(next_observation.last_detection)
        timestamp = float(next_observation.timestamp)

        self.belief_tracker.update_observation(
            scanned_bin=action_idx,
            detection=detection,
            timestamp=timestamp,
        )

        if detection:
            self.transition_estimator.record_detection(action_idx)

        self._last_processed_timestamp = timestamp
