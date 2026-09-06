"""rf_environment/scheduler/belief/state.py
Augmented Belief State representation over (Frequency, Dwell Phase, Dwell Duration).
"""

from __future__ import annotations

import copy
from typing import Any
import numpy as np

from rf_environment.scheduler.belief.config import BeliefSchedulerConfig


class AugmentedBeliefState:
    """Represents and updates the joint posterior distribution:
    b(F=f, tau=t, D=d) = P(F_t = f, tau_t = t, D = d | observations, actions)

    Properties:
    - Exact discrete distribution over (num_bins x sum(dwells)) states (e.g., 30 x 9 = 270 states).
    - Guarantees b >= 0 and sum(b) == 1.0 at all times.
    - Zero ground truth access; purely mathematical information state.
    """

    def __init__(
        self,
        num_bins: int,
        config: BeliefSchedulerConfig | None = None,
    ) -> None:
        self.num_bins = int(num_bins)
        self.config = config or BeliefSchedulerConfig()

        if self.config.ablation_mode == "B0":
            # B0: Frequency only (dwell fixed to 1)
            self.dwells: tuple[int, ...] = (1,)
        else:
            self.dwells = tuple(self.config.supported_dwells)

        # Total discrete states
        self.total_states = sum(self.num_bins * d for d in self.dwells)

        # Storage: dictionary of 2D arrays, shape (num_bins, d) for each d in self.dwells
        self.belief: dict[int, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        """Initializes belief state to an unbiased prior distribution across dwell hypotheses."""
        num_dwell_models = float(len(self.dwells))
        dwell_prior = 1.0 / num_dwell_models
        for d in self.dwells:
            state_prior = dwell_prior / float(self.num_bins * d)
            self.belief[d] = np.full((self.num_bins, d), state_prior, dtype=np.float64)
        self._normalize()

    def _normalize(self) -> None:
        """Clamps numerical underflow and strictly normalizes sum(b) == 1.0."""
        floor = self.config.min_probability_floor
        total = 0.0

        for d in self.dwells:
            np.maximum(self.belief[d], floor, out=self.belief[d])
            total += float(np.sum(self.belief[d]))

        if total <= 0.0 or not np.isfinite(total):
            # Degenerate fallback: reset to uniform
            prior_val = 1.0 / float(self.total_states)
            for d in self.dwells:
                self.belief[d].fill(prior_val)
            return

        inv_total = 1.0 / total
        for d in self.dwells:
            self.belief[d] *= inv_total

    def get_marginal_frequency(self) -> np.ndarray:
        """Returns marginal probability vector P(F = f) across all bins f in {0..num_bins-1}."""
        p_f = np.zeros(self.num_bins, dtype=np.float64)
        for d in self.dwells:
            p_f += np.sum(self.belief[d], axis=1)
        return p_f

    def get_marginal_dwell(self) -> dict[int, float]:
        """Returns marginal probability P(D = d) for each candidate dwell value."""
        return {d: float(np.sum(self.belief[d])) for d in self.dwells}

    def get_marginal_dwell_phase(self) -> dict[int, float]:
        """Returns marginal probability distribution over elapsed dwell tau."""
        max_d = max(self.dwells)
        phase_probs: dict[int, float] = {tau: 0.0 for tau in range(max_d)}
        for d in self.dwells:
            for tau in range(d):
                phase_probs[tau] += float(np.sum(self.belief[d][:, tau]))
        return phase_probs

    def update_observation(
        self,
        scanned_bin: int,
        detection: bool,
    ) -> None:
        """Performs Bayesian posterior update given detector outcome Y_t in {0, 1} on scanned bin A_t:
        P(Y | f = A_t) = P_D if Y=1 else (1 - P_D)
        P(Y | f != A_t) = P_FA if Y=1 else (1 - P_FA)
        """
        p_d = self.config.p_detection
        p_fa = self.config.p_false_alarm

        if detection:
            l_in_band = p_d
            l_out_band = p_fa
        else:
            l_in_band = 1.0 - p_d
            l_out_band = 1.0 - p_fa

        for d in self.dwells:
            arr = self.belief[d]
            # Multiply all channels except scanned_bin by l_out_band
            arr *= l_out_band
            # Multiply scanned_bin row by (l_in_band / l_out_band) to set it to l_in_band
            arr[scanned_bin, :] *= (l_in_band / l_out_band)

        self._normalize()

    def predict_transition(self, transition_matrix: np.ndarray) -> None:
        """Propagates belief state forward by one time step according to semi-Markov dwell dynamics:
        - If tau < d - 1: shifts mass from tau to tau + 1 (frequency unchanged).
        - If tau == d - 1: transitions mass to tau = 0 across frequencies using transition_matrix T.
        """
        new_belief: dict[int, np.ndarray] = {}

        for d in self.dwells:
            arr = self.belief[d]
            new_arr = np.zeros_like(arr)

            if d > 1:
                # Mass that has not finished dwell advances to tau + 1 on same frequency
                new_arr[:, 1:] = arr[:, :-1]

                # Mass at hop boundary tau = d - 1 transitions to new frequencies at tau = 0
                mass_out = arr[:, d - 1]  # shape (num_bins,)
                mass_in = mass_out @ transition_matrix  # shape (num_bins,)
                new_arr[:, 0] += mass_in
            else:
                # d == 1: hops every step
                mass_out = arr[:, 0]
                new_arr[:, 0] = mass_out @ transition_matrix

            new_belief[d] = new_arr

        self.belief = new_belief
        self._normalize()

    def get_entropy(self) -> float:
        """Returns Shannon entropy of the marginal frequency distribution (nats)."""
        p = self.get_marginal_frequency()
        p = p[p > 1e-12]
        return float(-np.sum(p * np.log(p)))

    def copy(self) -> AugmentedBeliefState:
        """Returns an independent deep copy of the belief state."""
        new_state = AugmentedBeliefState(self.num_bins, config=self.config)
        new_state.belief = {d: self.belief[d].copy() for d in self.dwells}
        return new_state
