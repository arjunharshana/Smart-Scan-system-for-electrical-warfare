from __future__ import annotations

from typing import Sequence
import numpy as np


class ReplayBuffer:
    """Experience Replay Buffer for Reinforcement Learning Schedulers.

    Stores transition tuples:
        (encoded_state, action, reward, encoded_next_state, done)

    Architectural Invariants:
    1. Contains zero ground-truth data (no EmitterState, true frequency, opportunity,
       coverage, interception, or future hop information).
    2. Stores strictly legitimate scheduler inputs and outcomes.
    3. Uniform random minibatch sampling with reproducible random seed.
    """

    def __init__(self, capacity: int = 10000, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)

        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []
        self.dones: list[bool] = []
        self.cursor: int = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Appends a transition tuple to the ring buffer."""
        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)

        if len(self.states) < self.capacity:
            self.states.append(state_arr)
            self.actions.append(int(action))
            self.rewards.append(float(reward))
            self.next_states.append(next_state_arr)
            self.dones.append(bool(done))
        else:
            self.states[self.cursor] = state_arr
            self.actions[self.cursor] = int(action)
            self.rewards[self.cursor] = float(reward)
            self.next_states[self.cursor] = next_state_arr
            self.dones[self.cursor] = bool(done)

        self.cursor = (self.cursor + 1) % self.capacity

    def sample(
        self,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples a random batch of transitions.

        Returns:
            states: [B, D] float32
            actions: [B] int64
            rewards: [B] float32
            next_states: [B, D] float32
            dones: [B] float32 (1.0 for True, 0.0 for False)
        """
        n = len(self.states)
        if n == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer.")

        k = min(int(batch_size), n)
        indices = self.rng.choice(n, size=k, replace=False)

        b_states = np.array([self.states[i] for i in indices], dtype=np.float32)
        b_actions = np.array([self.actions[i] for i in indices], dtype=np.int64)
        b_rewards = np.array([self.rewards[i] for i in indices], dtype=np.float32)
        b_next_states = np.array([self.next_states[i] for i in indices], dtype=np.float32)
        b_dones = np.array([1.0 if self.dones[i] else 0.0 for i in indices], dtype=np.float32)

        return b_states, b_actions, b_rewards, b_next_states, b_dones

    def __len__(self) -> int:
        return len(self.states)
