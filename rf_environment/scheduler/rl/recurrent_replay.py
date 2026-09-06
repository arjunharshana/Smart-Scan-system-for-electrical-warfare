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
        use_stored_hidden: bool = False,
        regime_balanced: bool = False,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if burn_in < 0:
            raise ValueError(f"burn_in must be non-negative, got {burn_in}")

        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        self.total_len = self.sequence_length + self.burn_in
        self.use_stored_hidden = bool(use_stored_hidden)
        self.regime_balanced = bool(regime_balanced)
        self.rng = np.random.default_rng(seed)

        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.next_states: list[np.ndarray] = []
        self.dones: list[bool] = []
        self.episode_ids: list[int] = []
        self.stored_h: list[np.ndarray | None] = []
        self.stored_c: list[np.ndarray | None] = []
        self.regime_ids: list[int | None] = []

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
        h: np.ndarray | None = None,
        c: np.ndarray | None = None,
        regime_id: int | None = None,
    ) -> None:
        """Appends transition to buffer."""
        state_arr = np.asarray(state, dtype=np.float32)
        next_arr = np.asarray(next_state, dtype=np.float32)
        h_arr = np.asarray(h, dtype=np.float32) if h is not None else None
        c_arr = np.asarray(c, dtype=np.float32) if c is not None else None
        r_id = int(regime_id) if regime_id is not None else None

        if len(self.states) < self.capacity:
            self.states.append(state_arr)
            self.actions.append(int(action))
            self.rewards.append(float(reward))
            self.next_states.append(next_arr)
            self.dones.append(bool(done))
            self.episode_ids.append(self.current_episode)
            self.stored_h.append(h_arr)
            self.stored_c.append(c_arr)
            self.regime_ids.append(r_id)
        else:
            self.states[self.cursor] = state_arr
            self.actions[self.cursor] = int(action)
            self.rewards[self.cursor] = float(reward)
            self.next_states[self.cursor] = next_arr
            self.dones[self.cursor] = bool(done)
            self.episode_ids[self.cursor] = self.current_episode
            self.stored_h[self.cursor] = h_arr
            self.stored_c[self.cursor] = c_arr
            self.regime_ids[self.cursor] = r_id

        self.cursor = (self.cursor + 1) % self.capacity

        if done:
            self.start_new_episode()

    def get_valid_start_indices(self) -> list[int]:
        """Identifies all start indices i where [i, i + total_len - 1] belong to the same episode
        and do not cross wrap-around or early-termination boundaries.
        """
        n = len(self.states)
        L = self.total_len
        if n < L:
            return []

        is_full = (n == self.capacity)
        valid = []
        for i in range(n - L + 1):
            # Circular buffer wrap-around fence: cursor must not fall inside (i, i + L - 1]
            if is_full and (i < self.cursor <= i + L - 1):
                continue
            ep_start = self.episode_ids[i]
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
        return_hidden: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None
    ]:
        """Samples a batch of contiguous sub-sequences of length total_len.

        Returns:
            If return_hidden is False:
                (b_states, b_actions, b_rewards, b_next_states, b_dones)
            If return_hidden is True:
                (b_states, b_actions, b_rewards, b_next_states, b_dones, b_h0, b_c0)
        """
        valid_starts = self.get_valid_start_indices()
        if not valid_starts:
            raise ValueError(f"Insufficient contiguous transitions (need {self.total_len}, have {len(self.states)}).")

        B = int(batch_size)
        if self.regime_balanced:
            # Group valid start indices by regime_id
            regime_groups: dict[Any, list[int]] = {}
            for idx in valid_starts:
                r_id = self.regime_ids[idx]
                if r_id not in regime_groups:
                    regime_groups[r_id] = []
                regime_groups[r_id].append(idx)

            distinct_groups = sorted(regime_groups.keys(), key=lambda x: (x is None, x))
            G = len(distinct_groups)
            if G > 1:
                # Sample equally from each regime group
                chosen_starts_list: list[int] = []
                base_per_group = B // G
                remainder = B % G
                for g_idx, g_key in enumerate(distinct_groups):
                    count = base_per_group + (1 if g_idx < remainder else 0)
                    if count > 0:
                        g_starts = regime_groups[g_key]
                        replace = len(g_starts) < count
                        sampled = self.rng.choice(g_starts, size=count, replace=replace)
                        chosen_starts_list.extend(sampled.tolist())
                chosen_starts = np.array(chosen_starts_list, dtype=np.int64)
                self.rng.shuffle(chosen_starts)
            else:
                k = min(B, len(valid_starts))
                chosen_starts = self.rng.choice(valid_starts, size=k, replace=(k < B))
        else:
            k = min(B, len(valid_starts))
            chosen_starts = self.rng.choice(valid_starts, size=k, replace=(k < B))

        L = self.total_len
        b_states = np.array([[self.states[i + j] for j in range(L)] for i in chosen_starts], dtype=np.float32)
        b_actions = np.array([[self.actions[i + j] for j in range(L)] for i in chosen_starts], dtype=np.int64)
        b_rewards = np.array([[self.rewards[i + j] for j in range(L)] for i in chosen_starts], dtype=np.float32)
        b_next_states = np.array([[self.next_states[i + j] for j in range(L)] for i in chosen_starts], dtype=np.float32)
        b_dones = np.array([[1.0 if self.dones[i + j] else 0.0 for j in range(L)] for i in chosen_starts], dtype=np.float32)

        if not return_hidden:
            return b_states, b_actions, b_rewards, b_next_states, b_dones

        b_h0: np.ndarray | None = None
        b_c0: np.ndarray | None = None
        if self.use_stored_hidden:
            h_list = [self.stored_h[i] for i in chosen_starts]
            c_list = [self.stored_c[i] for i in chosen_starts]
            if not any(h is None for h in h_list) and not any(c is None for c in c_list):
                b_h0 = np.array(h_list, dtype=np.float32)
                b_c0 = np.array(c_list, dtype=np.float32)

        return b_states, b_actions, b_rewards, b_next_states, b_dones, b_h0, b_c0

    def __len__(self) -> int:
        return len(self.states)

    @property
    def size(self) -> int:
        return len(self.states)
