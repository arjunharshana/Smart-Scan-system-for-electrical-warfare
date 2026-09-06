"""rf_environment/scheduler/belief/scheduler.py
V5.0 Augmented Belief-State Bayesian Scheduler for Partially Observable RF Scanning.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler
from rf_environment.scheduler.belief.config import BeliefSchedulerConfig
from rf_environment.scheduler.belief.state import AugmentedBeliefState
from rf_environment.scheduler.belief.transition import ObservableTransitionModel


class V5BeliefScheduler(BaseScheduler):
    """V5.0 Augmented Belief-State Bayesian Scheduler.

    Formulates electronic warfare spectrum scanning as a partially observable controlled process.
    Maintains an exact recursive Bayesian belief state:
        b_t(F, tau, D) = P(F_t = F, tau_t = tau, D_t = D | observations, actions)
    across candidate frequencies F in {0..N-1}, dwell phases tau in {0..D-1}, and candidate
    dwell durations D in {1, 3, 5}.

    Key Invariants:
    1. ZERO Neural Networks, ZERO RL, ZERO offline training, ZERO online gradient updates.
    2. STRICT Ground-Truth Isolation: Operates solely on canonical SchedulerObservation.
    3. Closed-Form Bayesian Updates: Exact sensor likelihood update using detector (P_D=0.95, P_FA=0.02).
    4. Exact Semi-Markov Progression: Propagates dwell duration and hop boundaries forward in time.
    5. Sub-millisecond CPU Latency: Evaluates full 270-state distribution in < 100 microseconds.
    """

    name: str = "v5_belief"
    category: str = "bayesian_pomdp"

    def __init__(
        self,
        bands_hz: list[float],
        config: BeliefSchedulerConfig | dict[str, Any] | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(bands_hz)
        self.num_bins = len(self.bands_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if isinstance(config, BeliefSchedulerConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = BeliefSchedulerConfig.from_dict(config)
        else:
            cfg_dict = dict(kwargs)
            self.config = BeliefSchedulerConfig.from_dict(cfg_dict)

        self.belief_state = AugmentedBeliefState(self.num_bins, config=self.config)
        self.transition_model = ObservableTransitionModel(self.num_bins, config=self.config)

        self._time_since_scan = np.zeros(self.num_bins, dtype=np.float64)
        self._last_processed_timestamp: float | None = None
        self._step_count: int = 0

    def reset(self) -> None:
        """Resets belief state, transition models, and scan timers with zero cross-episode leakage."""
        super().reset()
        self.belief_state.reset()
        self.transition_model.reset()
        self._time_since_scan.fill(0.0)
        self._last_processed_timestamp = None
        self._step_count = 0

    def select_bin(self, observation: SchedulerObservation | Any = None) -> int:
        """Processes previous observation and selects the next frequency bin."""
        if isinstance(observation, SchedulerObservation):
            # Process transition if observation is newer than last processed timestamp
            if (
                self._last_processed_timestamp is None
                or observation.timestamp > self._last_processed_timestamp
            ):
                if self._last_processed_timestamp is not None:
                    # Previous action outcome
                    prev_bin = observation.current_frequency_bin
                    prev_det = bool(observation.last_detection)

                    # 1. Recursive Bayesian likelihood update: P(Y | s, A)
                    self.belief_state.update_observation(
                        scanned_bin=prev_bin,
                        detection=prev_det,
                    )

                    # 2. Record detection in empirical transition tracker (B2 mode)
                    if prev_det:
                        self.transition_model.record_detection(
                            frequency_bin=prev_bin,
                            timestamp=observation.timestamp,
                        )

                    # 3. Semi-Markov forward temporal propagation: b_{t+1}^- = T(b_t)
                    t_matrix = self.transition_model.get_transition_matrix()
                    self.belief_state.predict_transition(t_matrix)

                self._last_processed_timestamp = observation.timestamp

        # Update coverage/staleness timers
        self._time_since_scan += 1.0
        if self.last_selected_bin is not None and 0 <= self.last_selected_bin < self.num_bins:
            self._time_since_scan[self.last_selected_bin] = 0.0

        # Marginal channel probability: P(F_t = f) = sum_{d, tau} b(f, tau, d)
        p_f = self.belief_state.get_marginal_frequency()

        # Action selection score
        if self.config.exploration_mode == "uncertainty":
            # Principled coverage bonus based on scan staleness
            u_f = np.minimum(1.0, self._time_since_scan / float(self.config.staleness_threshold))
            scores = p_f + float(self.config.uncertainty_weight) * u_f
        else:
            scores = p_f

        # Deterministic tie-breaking: select highest score; break ties to lowest bin index
        # Adding tiny deterministic descending penalty for strict tie-breaking
        tie_breaker = np.linspace(1e-12, 0.0, self.num_bins)
        selected_bin = int(np.argmax(scores + tie_breaker))
        self.last_selected_bin = selected_bin
        self.last_scores = scores


        # Record diagnostic telemetry
        dwell_marginal = self.belief_state.get_marginal_dwell()
        most_likely_dwell = max(dwell_marginal.keys(), key=lambda d: dwell_marginal[d])

        self.last_explanation = {
            "scheduler": self.name,
            "ablation_mode": self.config.ablation_mode,
            "selected_bin": selected_bin,
            "max_belief": float(p_f[selected_bin]),
            "inferred_dwell": int(most_likely_dwell),
            "dwell_probabilities": dwell_marginal,
            "entropy": float(self.belief_state.get_entropy()),
            "score": float(scores[selected_bin]),
        }
        self._step_count += 1

        return selected_bin
