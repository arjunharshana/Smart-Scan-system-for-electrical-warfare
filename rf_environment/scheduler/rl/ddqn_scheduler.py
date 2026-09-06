from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler
from rf_environment.scheduler.rl.encoder import ObservationEncoder
from rf_environment.scheduler.rl.network import MLPQNetwork
from rf_environment.scheduler.rl.replay_buffer import ReplayBuffer


class DDQNScheduler(BaseScheduler):
    """Double Deep Q-Network (DDQN) Scan Scheduler for Electronic Warfare Interception.

    Architectural Invariants:
    1. Consumes strictly canonical SchedulerObservation (zero ground truth).
    2. Emits strictly canonical ScanAction(frequency_bin).
    3. Uses exact Double DQN target:
           y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))
    4. Separates online and target networks; updates target periodically.
    5. Clean train / eval mode separation (evaluation uses deterministic greedy policy).
    6. Seed-reproducible exploration, network initialization, and replay sampling.
    """

    name = "ddqn"
    category = "rl"

    def __init__(
        self,
        bands_hz: list[float],
        gamma: float = 0.95,
        learning_rate: float = 0.001,
        replay_capacity: int = 10000,
        batch_size: int = 32,
        warmup_steps: int = 64,
        target_update_frequency: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        hidden_dimension: int = 64,
        encoder: ObservationEncoder | None = None,
        seed: int | None = None,
        checkpoint_path: str | Path | None = None,
        require_checkpoint: bool = False,
    ) -> None:
        super().__init__(bands_hz)
        self.num_bins = len(self.bands_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Hyperparameters
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.replay_capacity = int(replay_capacity)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.target_update_frequency = int(target_update_frequency)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = float(epsilon_start)
        self.hidden_dimension = int(hidden_dimension)

        # Observation Encoder
        self.encoder = encoder or ObservationEncoder(num_bins=self.num_bins)

        # Neural Networks
        net_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        self.online_net = MLPQNetwork(
            input_dim=self.encoder.feature_dim,
            output_dim=self.num_bins,
            hidden_dim=self.hidden_dimension,
            learning_rate=self.learning_rate,
            seed=net_seed,
        )
        self.target_net = self.online_net.clone()

        # Replay Buffer
        buffer_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        self.replay_buffer = ReplayBuffer(
            capacity=self.replay_capacity,
            seed=buffer_seed,
        )

        # Training telemetry
        self.train_step_count: int = 0
        self.last_loss: float = 0.0
        self.losses: list[float] = []

        # Checkpoint state
        self.checkpoint_path: str | None = None
        self.checkpoint_sha256: str | None = None
        self.is_pretrained: bool = False

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        elif require_checkpoint:
            raise FileNotFoundError(
                "V4.0 CHECKPOINT NOT FOUND — INFERENCE UNAVAILABLE\n"
                "Checkpoint path was not provided and require_checkpoint=True.\n"
                "Refusing to start with random/untrained weights."
            )

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Serializes current model parameters into a versioned .npz checkpoint."""
        from rf_environment.scheduler.rl.checkpoint_v40 import save_v40_checkpoint

        saved = save_v40_checkpoint(self, path, metadata=metadata)
        self.checkpoint_path = str(saved)
        return saved

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Loads weights from a validated checkpoint and freezes inference."""
        from rf_environment.scheduler.rl.checkpoint_v40 import load_v40_checkpoint

        ckpt_data = load_v40_checkpoint(
            path=path,
            expected_input_dim=self.encoder.feature_dim,
            expected_output_dim=self.num_bins,
            expected_bands_hz=self.bands_hz,
        )

        arch = ckpt_data["metadata"].get("architecture", {})
        ckpt_hidden = arch.get("hidden_dim")
        if ckpt_hidden is not None and ckpt_hidden != self.hidden_dimension:
            self.hidden_dimension = int(ckpt_hidden)
            self.online_net = MLPQNetwork(
                input_dim=self.encoder.feature_dim,
                output_dim=self.num_bins,
                hidden_dim=self.hidden_dimension,
                learning_rate=self.learning_rate,
            )
            self.target_net = self.online_net.clone()

        self.online_net.load_weights_dict(ckpt_data["online_weights"])
        self.target_net.load_weights_dict(ckpt_data["target_weights"])

        # Production frozen inference invariants
        self.eval()
        self.epsilon = 0.0

        self.checkpoint_path = str(ckpt_data["path"])
        self.checkpoint_sha256 = str(ckpt_data["weights_sha256"])
        self.is_pretrained = True
        return ckpt_data

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        bands_hz: list[float] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> DDQNScheduler:
        """Constructs a DDQNScheduler directly from a saved checkpoint."""
        from rf_environment.scheduler.rl.checkpoint_v40 import load_v40_checkpoint

        ckpt = load_v40_checkpoint(path)
        meta = ckpt["metadata"]
        saved_bands = [float(b) for b in meta.get("bands_hz", [])]
        resolved_bands = bands_hz or saved_bands
        if not resolved_bands:
            raise ValueError("No frequency bands available in checkpoint or arguments.")

        arch = meta.get("architecture", {})
        hidden_dim = arch.get("hidden_dim", 64)

        scheduler = cls(
            bands_hz=resolved_bands,
            hidden_dimension=hidden_dim,
            seed=seed,
            checkpoint_path=path,
            **kwargs,
        )
        return scheduler

    def get_q_values(self, observation: SchedulerObservation) -> np.ndarray:
        """Evaluates and returns raw Q-values for all frequency bins from online network."""
        state_vec = self.encoder.encode(observation)
        return self.online_net.forward(state_vec)

    def select_bin(self, observation: SchedulerObservation) -> int:
        """Selects frequency bin via epsilon-greedy exploration in train mode or greedy in eval."""
        q_values = self.get_q_values(observation)

        if self.train_mode and self.rng.random() < self.epsilon:
            # Epsilon exploration
            bin_idx = int(self.rng.integers(0, self.num_bins))
            reason = f"DDQN Epsilon Exploration (eps={self.epsilon:.3f})"
        else:
            # Greedy exploitation
            bin_idx = int(np.argmax(q_values))
            reason = f"DDQN Greedy Exploitation Q={q_values[bin_idx]:.3f}"

        # Decay epsilon on training decisions
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
            "rule": "ddqn",
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
        """Stores transition and executes Double DQN parameter optimization."""
        if not self.train_mode:
            return

        state = self.encoder.encode(observation)
        next_state = self.encoder.encode(next_observation)
        action_idx = action.frequency_bin if isinstance(action, ScanAction) else int(action)

        # 1. Store transition in replay buffer (zero ground truth)
        self.replay_buffer.push(state, action_idx, reward, next_state, done)

        # 2. Train once warmup threshold is met
        if len(self.replay_buffer) >= self.warmup_steps:
            loss = self._train_step()
            self.last_loss = loss
            self.losses.append(loss)

    def _train_step(self) -> float:
        """Executes a single Double DQN minibatch gradient update."""
        b_states, b_actions, b_rewards, b_next_states, b_dones = self.replay_buffer.sample(self.batch_size)

        # Double DQN Target:
        # a* = argmax_a Q_online(s', a)
        next_q_online = self.online_net.forward(b_next_states)  # [B, N]
        best_actions = np.argmax(next_q_online, axis=1)          # [B]

        # Target value evaluated by target network at a*:
        next_q_target = self.target_net.forward(b_next_states)  # [B, N]
        target_q_values = next_q_target[np.arange(len(b_states)), best_actions]  # [B]

        # Bellman target: y = r + gamma * (1 - done) * Q_target(s', a*)
        targets = b_rewards + self.gamma * (1.0 - b_dones) * target_q_values

        # Optimize online network
        loss = self.online_net.train_step(b_states, b_actions, targets)

        self.train_step_count += 1

        # Periodic target network synchronization
        if self.train_step_count % self.target_update_frequency == 0:
            self.target_net.copy_from(self.online_net)

        return loss

    def reset(self) -> None:
        super().reset()
        self.epsilon = self.epsilon_start if self.train_mode else 0.0
