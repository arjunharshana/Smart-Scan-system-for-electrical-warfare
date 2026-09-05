from __future__ import annotations

from typing import Sequence
import numpy as np

from rf_environment.scheduler.hybrid.continual.config import V42ContinualConfig
from rf_environment.scheduler.rl.recurrent_replay import RecurrentReplayBuffer


class OnlineReplayBuffer:
    """Continual learning experience replay buffer with mixed recent and historical sampling.

    Maintains two separate sequence reservoirs to prevent catastrophic forgetting:
    1. Recent Buffer: Captures live online experiences from newly observed RF regimes.
    2. Historical Buffer: Retains protected representative trajectories from baseline/prior regimes.

    Invariants:
    1. Zero ground truth stored (only encoded state vectors, actions, observable rewards).
    2. Sequence integrity: sampled sub-trajectories do not cross episode boundaries.
    3. Configurable mixing ratio (default: 70% recent, 30% historical).
    """

    def __init__(
        self,
        config: V42ContinualConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or V42ContinualConfig()
        self.rng = np.random.default_rng(seed)

        seed_recent = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        seed_hist = self.rng.integers(0, 2**31 - 1) if seed is not None else None

        self.recent_buffer = RecurrentReplayBuffer(
            capacity=self.config.recent_replay_capacity,
            sequence_length=self.config.sequence_length,
            burn_in=self.config.burn_in,
            seed=seed_recent,
        )

        self.historical_buffer = RecurrentReplayBuffer(
            capacity=self.config.historical_replay_capacity,
            sequence_length=self.config.sequence_length,
            burn_in=self.config.burn_in,
            seed=seed_hist,
        )

        self.sequence_length = self.config.sequence_length
        self.recent_ratio = self.config.recent_replay_ratio

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Appends live online transition to recent buffer."""
        self.recent_buffer.push(state, action, reward, next_state, done)

    def push_historical(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stores a transition in the protected historical buffer."""
        self.historical_buffer.push(state, action, reward, next_state, done)

    def populate_historical_from_buffer(self, source_buffer: RecurrentReplayBuffer) -> None:
        """Copies existing baseline transitions into the historical reservoir."""
        n = len(source_buffer.states)
        for i in range(n):
            self.historical_buffer.push(
                source_buffer.states[i],
                source_buffer.actions[i],
                source_buffer.rewards[i],
                source_buffer.next_states[i],
                source_buffer.dones[i],
            )

    def has_min_sequences(self, min_count: int | None = None) -> bool:
        """Checks if recent buffer has enough valid contiguous sequences for adaptation."""
        target = min_count if min_count is not None else self.config.min_replay_sequences
        valid_recent = self.recent_buffer.get_valid_start_indices()
        return len(valid_recent) >= target

    def sample_mixed(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples a mixed batch of contiguous sequences (recent + historical)."""
        valid_recent = len(self.recent_buffer.get_valid_start_indices())
        valid_hist = len(self.historical_buffer.get_valid_start_indices())

        if valid_recent == 0 and valid_hist == 0:
            raise ValueError("Cannot sample from empty replay buffer.")

        # Determine allocation
        if valid_hist == 0:
            # Fallback to pure recent if no historical data exists
            return self.recent_buffer.sample(batch_size)
        elif valid_recent == 0:
            # Fallback to pure historical if no recent data exists
            return self.historical_buffer.sample(batch_size)

        n_recent = max(1, int(round(batch_size * self.recent_ratio)))
        n_hist = max(1, batch_size - n_recent)

        # Sample from each buffer
        b_rec = self.recent_buffer.sample(n_recent)
        b_hst = self.historical_buffer.sample(n_hist)

        # Concatenate along batch dimension
        states = np.concatenate([b_rec[0], b_hst[0]], axis=0)
        actions = np.concatenate([b_rec[1], b_hst[1]], axis=0)
        rewards = np.concatenate([b_rec[2], b_hst[2]], axis=0)
        next_states = np.concatenate([b_rec[3], b_hst[3]], axis=0)
        dones = np.concatenate([b_rec[4], b_hst[4]], axis=0)

        return states, actions, rewards, next_states, dones

    def sample_recent_validation(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples sequences strictly from recent buffer for new-regime validation."""
        return self.recent_buffer.sample(batch_size)

    def sample_historical_validation(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Samples sequences strictly from historical buffer for retention validation."""
        valid_hist = len(self.historical_buffer.get_valid_start_indices())
        if valid_hist < self.config.sequence_length:
            return None
        return self.historical_buffer.sample(batch_size)

    def reset(self) -> None:
        """Clears recent buffer while preserving protected historical buffer."""
        self.recent_buffer = RecurrentReplayBuffer(
            capacity=self.config.recent_replay_capacity,
            sequence_length=self.config.sequence_length,
            burn_in=self.config.burn_in,
        )

    def reset_all(self) -> None:
        """Clears both recent and historical buffers."""
        self.recent_buffer = RecurrentReplayBuffer(
            capacity=self.config.recent_replay_capacity,
            sequence_length=self.config.sequence_length,
            burn_in=self.config.burn_in,
        )
        self.historical_buffer = RecurrentReplayBuffer(
            capacity=self.config.historical_replay_capacity,
            sequence_length=self.config.sequence_length,
            burn_in=self.config.burn_in,
        )
