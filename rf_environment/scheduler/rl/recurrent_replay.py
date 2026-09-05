from __future__ import annotations

from typing import Sequence
import numpy as np


class RecurrentReplayBuffer:
    """Experience Replay Buffer for Recurrent (LSTM / DRQN) RL Schedulers.

    Stores sequential transitions and samples contiguous sub-sequences of length L:
        (states_seq, actions_seq, rewards_seq, next_states_seq, dones_seq)

    Architectural Invariants:
    1. Zero ground truth (no true emitter parameters, true frequencies, or opportunity labels).
    2. Sequence integrity: sampled sub-trajectories do not cross episode boundaries.
    3. Reproducible uniform random sampling controlled by seed.
    4. Configurable sequence length and burn-in.
    """

    def __init__(
        self,
        capacity: int = 10000,
        sequence_length: int = 10,
        burn_in: int = 0,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")

        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        self.rng = np.random.default_rng(seed)

        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []
        self.dones: list[bool] = []
        self.episode_ids: list[int] = []

        self.current_episode: int = 0
        self.cursor: int = 0

    def start_new_episode(self) -> None:
        """Increments episode counter to ensure sequences never cross episodes."""
        self.current_episode += 1

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Appends transition to buffer."""
        state_arr = np.asarray(state, dtype=np.float32)
        next_arr = np.asarray(next_state, dtype=np.float32)

        if len(self.states) < self.capacity:
            self.states.append(state_arr)
            self.actions.append(int(action))
            self.rewards.append(float(reward))
            self.next_states.append(next_arr)
            self.dones.append(bool(done))
            self.episode_ids.append(self.current_episode)
        else:
            self.states[self.cursor] = state_arr
            self.actions[self.cursor] = int(action)
            self.rewards[self.cursor] = float(reward)
            self.next_states[self.cursor] = next_arr
            self.dones[self.cursor] = bool(done)
            self.episode_ids[self.cursor] = self.current_episode

        self.cursor = (self.cursor + 1) % self.capacity

        if done:
            self.start_new_episode()

    def get_valid_start_indices(self) -> list[int]:
        """Identifies all start indices i where [i, i + L - 1] belong to the same episode."""
        n = len(self.states)
        L = self.sequence_length
        if n < L:
            return []

        valid = []
        for i in range(n - L + 1):
            ep_start = self.episode_ids[i]
            # Verify no episode boundary is crossed within the sequence
            same_ep = True
            for k in range(1, L):
                if self.episode_ids[i + k] != ep_start:
                    same_ep = False
                    break
                # If a done occurs before the end of the window, sequence ends early
                if self.dones[i + k - 1]:
                    same_ep = False
                    break
            if same_ep:
                valid.append(i)
        return valid

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples a batch of contiguous sub-sequences of length L.

        Returns:
            b_states: [B, L, D] float32
            b_actions: [B, L] int64
            b_rewards: [B, L] float32
            b_next_states: [B, L, D] float32
            b_dones: [B, L] float32
        """
        valid_starts = self.get_valid_start_indices()
        if not valid_starts:
            raise ValueError(f"Insufficient contiguous transitions (need {self.sequence_length}, have {len(self.states)}).")

        k = min(int(batch_size), len(valid_starts))
        chosen_starts = self.rng.choice(valid_starts, size=k, replace=(k < batch_size))

        L = self.sequence_length
        b_states = np.array([[self.states[i + j] for j in range(L)] for i in chosen_starts], dtype=np.float32)
        b_actions = np.array([[self.actions[i + j] for j in range(L)] for i in chosen_starts], dtype=np.int64)
        b_rewards = np.array([[self.rewards[i + j] for j in range(L)] for i in chosen_starts], dtype=np.float32)
        b_next_states = np.array([[self.next_states[i + j] for j in range(L)] for i in chosen_starts], dtype=np.float32)
        b_dones = np.array([[1.0 if self.dones[i + j] else 0.0 for j in range(L)] for i in chosen_starts], dtype=np.float32)

        return b_states, b_actions, b_rewards, b_next_states, b_dones

    def __len__(self) -> int:
        return len(self.states)

    @property
    def size(self) -> int:
        return len(self.states)
