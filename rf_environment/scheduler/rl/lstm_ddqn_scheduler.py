from __future__ import annotations

from typing import Any
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler
from rf_environment.scheduler.rl.encoder import ObservationEncoder
from rf_environment.scheduler.rl.lstm_network import LSTMQNetwork
from rf_environment.scheduler.rl.recurrent_replay import RecurrentReplayBuffer
from rf_environment.scheduler.rl.temporal_encoder import TemporalObservationEncoder
from rf_environment.util.seeding import derive_seed


class LSTMDDQNScheduler(BaseScheduler):
    """V4.1 Recurrent Double DQN Scan Scheduler with Working Memory (LSTM).

    Replaces feed-forward MLP with a genuine Recurrent Q-Network (DRQN) maintaining
    internal temporal hidden state (h_t, c_t) across successive scanning steps.

    Architectural Invariants:
    1. Consumes strictly canonical SchedulerObservation (zero ground truth).
    2. Emits strictly canonical ScanAction(frequency_bin) over common action space {0..N-1}.
    3. Maintains hidden-state lifecycle: reset starts a fresh (h_0 = 0, c_0 = 0).
       Zero hidden-state leakage between episodes or evaluation runs.
    4. Sequence-based Double DQN Bellman optimization with target network separation.
    5. Pure NumPy implementation (zero external deep-learning dependencies).
    """

    name = "lstm_ddqn"
    category = "rl"

    def __init__(
        self,
        bands_hz: list[float],
        encoder: ObservationEncoder | None = None,
        hidden_dim: int = 64,
        dense_dim: int = 64,
        sequence_length: int = 10,
        burn_in: int = 0,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        replay_capacity: int = 10000,
        batch_size: int = 32,
        warmup_steps: int = 64,
        target_update_frequency: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
        checkpoint_path: str | Path | None = None,
        require_checkpoint: bool = False,
    ) -> None:
        super().__init__(bands_hz)
        self.num_bins = len(self.bands_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.checkpoint_path: str | None = None
        self.checkpoint_sha256: str | None = None
        self.is_pretrained: bool = False

        self.hidden_dim = int(hidden_dim)
        self.dense_dim = int(dense_dim)
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.target_update_frequency = int(target_update_frequency)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = self.epsilon_start

        # 1. Observation Encoder
        self.encoder = encoder or TemporalObservationEncoder.create_encoder_b(num_bins=self.num_bins)

        # 2. Recurrent Online & Target Networks
        net_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        self.online_net = LSTMQNetwork(
            input_dim=self.encoder.feature_dim,
            output_dim=self.num_bins,
            hidden_dim=self.hidden_dim,
            dense_dim=self.dense_dim,
            learning_rate=learning_rate,
            seed=net_seed,
        )
        self.target_net = self.online_net.clone()

        # 3. Recurrent Sequence Replay Buffer
        buf_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        self.replay_buffer = RecurrentReplayBuffer(
            capacity=replay_capacity,
            sequence_length=self.sequence_length,
            burn_in=self.burn_in,
            seed=buf_seed,
        )

        # 4. Hidden & Cell States Lifecycle
        self.h: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)
        self.c: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)

        # Step caching to prevent multiple advancements on the same observation
        self._last_obs_timestamp: float | None = None
        self._last_q_values: np.ndarray = np.zeros(self.num_bins, dtype=np.float32)

        # Telemetry
        self.train_step_count: int = 0
        self.last_loss: float = 0.0
        self.losses: list[float] = []

        # 5. Production Checkpoint Loading
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif require_checkpoint:
            raise FileNotFoundError(
                "V4.1 production checkpoint not found (checkpoint_path was not provided).\n"
                "Refusing to start with random/untrained LSTM weights."
            )

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Serializes current model parameters into a versioned .npz checkpoint."""
        from rf_environment.scheduler.rl.checkpoint import save_checkpoint

        saved = save_checkpoint(self, path, metadata=metadata)
        self.checkpoint_path = str(saved)
        return saved

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Loads weights from a validated checkpoint, resetting hidden state and freezing inference."""
        from rf_environment.scheduler.rl.checkpoint import load_checkpoint

        ckpt_data = load_checkpoint(
            path=path,
            expected_input_dim=self.encoder.feature_dim,
            expected_output_dim=self.num_bins,
            expected_bands_hz=self.bands_hz,
        )

        self.online_net.load_weights_dict(ckpt_data["online_weights"])
        self.target_net.load_weights_dict(ckpt_data["target_weights"])

        # Reset recurrent memory to clean zeros
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.c = np.zeros(self.hidden_dim, dtype=np.float32)

        # Production frozen inference invariants
        self.eval()
        self.epsilon = 0.0

        self.checkpoint_path = str(ckpt_data["path"])
        self.checkpoint_sha256 = str(ckpt_data["sha256"])
        self.is_pretrained = True
        return ckpt_data

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        bands_hz: list[float] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> LSTMDDQNScheduler:
        """Constructs an LSTMDDQNScheduler directly from a saved checkpoint."""
        from rf_environment.scheduler.rl.checkpoint import load_checkpoint

        ckpt_data = load_checkpoint(path=path)
        meta = ckpt_data["metadata"]
        arch = meta.get("architecture", {})
        loaded_bands = bands_hz or meta.get("bands_hz")
        if loaded_bands is None:
            raise ValueError("bands_hz must be provided or present in checkpoint metadata.")

        sched = cls(
            bands_hz=loaded_bands,
            hidden_dim=arch.get("hidden_dim", 64),
            dense_dim=arch.get("dense_dim", 64),
            sequence_length=arch.get("sequence_length", 10),
            seed=seed,
            checkpoint_path=path,
            require_checkpoint=True,
            **kwargs,
        )
        return sched

    def get_q_values(self, observation: SchedulerObservation) -> np.ndarray:
        """Updates internal recurrent state and returns Q-values for all frequency bins."""
        if (
            self._last_obs_timestamp is not None
            and observation.timestamp == self._last_obs_timestamp
        ):
            return np.copy(self._last_q_values)

        state_vec = self.encoder.encode(observation)
        q_val, self.h, self.c = self.online_net.step(state_vec, self.h, self.c)

        self._last_obs_timestamp = observation.timestamp
        self._last_q_values = np.copy(q_val)
        return np.copy(q_val)

    def get_hidden_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns copies of current hidden state (h, c)."""
        return np.copy(self.h), np.copy(self.c)

    def select_bin(self, observation: SchedulerObservation) -> int:
        """Selects frequency bin via epsilon-greedy in train mode or greedy in eval."""
        q_values = self.get_q_values(observation)

        if self.train_mode and self.rng.random() < self.epsilon:
            bin_idx = int(self.rng.integers(0, self.num_bins))
            reason = f"LSTM-DDQN Epsilon Exploration (eps={self.epsilon:.3f})"
        else:
            bin_idx = int(np.argmax(q_values))
            reason = f"LSTM-DDQN Greedy Exploitation Q={q_values[bin_idx]:.3f}"

        if self.train_mode:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.last_selected_bin = bin_idx
        self.last_selected = self.bands_hz[bin_idx]
        self.last_explanation = {
            "action_mhz": self.last_selected / 1e6,
            "q_values": q_values.tolist(),
            "selected_q": float(q_values[bin_idx]),
            "epsilon": float(self.epsilon),
            "reason": reason,
            "rule": "lstm_ddqn",
        }
        return bin_idx

    def update_policy(
        self,
        observation: SchedulerObservation,
        action: ScanAction | int,
        reward: float,
        next_observation: SchedulerObservation,
        done: bool,
    ) -> None:
        """Stores transition and executes Recurrent Double DQN parameter updates."""
        if not self.train_mode:
            return

        state = self.encoder.encode(observation)
        next_state = self.encoder.encode(next_observation)
        action_idx = action.frequency_bin if isinstance(action, ScanAction) else int(action)

        self.replay_buffer.push(state, action_idx, reward, next_state, done)

        # Train once warmup threshold is met and valid contiguous sequences exist
        if len(self.replay_buffer) >= self.warmup_steps:
            valid_starts = self.replay_buffer.get_valid_start_indices()
            if valid_starts:
                loss = self._train_step()
                self.last_loss = loss
                self.losses.append(loss)

    def _train_step(self) -> float:
        """Samples contiguous sequence minibatch and applies Recurrent Double DQN update."""
        b_states, b_actions, b_rewards, b_next_states, b_dones = self.replay_buffer.sample(self.batch_size)

        # 1. Double DQN Next Action Selection (Online Network)
        next_q_online, _, _ = self.online_net.forward_sequence(b_next_states)  # [B, L, N]
        best_next_actions = np.argmax(next_q_online, axis=2)                   # [B, L]

        # 2. Target Network Evaluation at a*
        next_q_target, _, _ = self.target_net.forward_sequence(b_next_states)  # [B, L, N]
        target_next_q = np.take_along_axis(
            next_q_target, best_next_actions[:, :, np.newaxis], axis=2
        ).squeeze(2)  # [B, L]

        # 3. Bellman Target: y = r + gamma * (1 - done) * Q_target(s', a*)
        targets = b_rewards + self.gamma * (1.0 - b_dones) * target_next_q

        # 4. Backpropagation Through Time on Online Network
        loss = self.online_net.train_step(
            b_states, b_actions, targets, burn_in=self.burn_in
        )

        self.train_step_count += 1

        # 5. Periodic Target Network Copy
        if self.train_step_count % self.target_update_frequency == 0:
            self.target_net.copy_from(self.online_net)

        return loss

    def reset(self) -> None:
        """Resets recurrent state and episode tracking (zero cross-episode leakage)."""
        super().reset()
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.c = np.zeros(self.hidden_dim, dtype=np.float32)
        self.epsilon = self.epsilon_start if self.train_mode else 0.0
        self._last_obs_timestamp = None
        self._last_q_values = np.zeros(self.num_bins, dtype=np.float32)
        self.replay_buffer.start_new_episode()

    def reseed(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        net_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        buf_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        self.online_net.rng = np.random.default_rng(net_seed)
        self.replay_buffer.rng = np.random.default_rng(buf_seed)
