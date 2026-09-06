from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.base import BaseScheduler
from rf_environment.scheduler.rl.encoder import ObservationEncoder
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
from rf_environment.scheduler.rl.recurrent_replay import RecurrentReplayBuffer
from rf_environment.scheduler.rl.temporal_encoder import TemporalObservationEncoder
from rf_environment.scheduler.rl.torch_lstm_network import TorchLSTMQNetwork


class TorchLSTMDDQNScheduler(BaseScheduler):
    """GPU-accelerated PyTorch Recurrent Double DQN Scan Scheduler.

    Executes offline training on CUDA (or CPU fallback if CUDA unavailable)
    using cuDNN-optimized LSTM and PyTorch autograd.
    Exports 100% mathematically equivalent NumPy .npz checkpoints for production inference.
    """

    name = "torch_lstm_ddqn"
    category = "rl"

    def __init__(
        self,
        bands_hz: list[float],
        encoder: ObservationEncoder | None = None,
        hidden_dim: int = 64,
        dense_dim: int = 64,
        sequence_length: int = 10,
        burn_in: int = 0,
        use_stored_hidden: bool = True,
        regime_balanced_replay: bool = False,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        replay_capacity: int = 10000,
        batch_size: int = 32,
        warmup_steps: int = 64,
        target_update_frequency: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        max_grad_norm: float = 5.0,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(bands_hz)
        self.num_bins = len(self.bands_hz)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Device placement
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.hidden_dim = int(hidden_dim)
        self.dense_dim = int(dense_dim)
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        self.use_stored_hidden = bool(use_stored_hidden)
        self.regime_balanced_replay = bool(regime_balanced_replay)
        self.current_regime_id: int | None = None
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.target_update_frequency = int(target_update_frequency)
        self.max_grad_norm = float(max_grad_norm)
        self.learning_rate = float(learning_rate)

        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = self.epsilon_start

        # 1. Observation Encoder
        self.encoder = encoder or TemporalObservationEncoder.create_encoder_b(num_bins=self.num_bins)

        # 2. Networks
        net_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        self.online_net = TorchLSTMQNetwork(
            input_dim=self.encoder.feature_dim,
            output_dim=self.num_bins,
            hidden_dim=self.hidden_dim,
            dense_dim=self.dense_dim,
            device=self.device,
            seed=net_seed,
        )
        self.target_net = self.online_net.clone()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate)

        # 3. Recurrent Replay Buffer
        buf_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        self.replay_buffer = RecurrentReplayBuffer(
            capacity=replay_capacity,
            sequence_length=self.sequence_length,
            burn_in=self.burn_in,
            use_stored_hidden=self.use_stored_hidden,
            regime_balanced=self.regime_balanced_replay,
            seed=buf_seed,
        )

        # 4. Hidden & Cell States Lifecycle (NumPy arrays for fast env interaction)
        self.h: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)
        self.c: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)
        self._prev_h: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)
        self._prev_c: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)

        # Step caching
        self._last_obs_timestamp: float | None = None
        self._last_q_values: np.ndarray = np.zeros(self.num_bins, dtype=np.float32)

        # Training Telemetry
        self.train_step_count: int = 0
        self.last_loss: float = 0.0
        self.losses: list[float] = []
        self.grad_norms: list[float] = []
        self.td_errors: list[float] = []
        self.q_means: list[float] = []
        self.q_vars: list[float] = []

    def get_q_values(self, observation: SchedulerObservation) -> np.ndarray:
        """Updates internal recurrent state via PyTorch step and returns Q-values."""
        if (
            self._last_obs_timestamp is not None
            and observation.timestamp == self._last_obs_timestamp
        ):
            return np.copy(self._last_q_values)

        state_vec = self.encoder.encode(observation)
        self._prev_h = np.copy(self.h)
        self._prev_c = np.copy(self.c)

        # Single step forward in PyTorch
        self.online_net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)
            h_t = torch.from_numpy(self.h).unsqueeze(0).to(self.device)
            c_t = torch.from_numpy(self.c).unsqueeze(0).to(self.device)
            q_val, h_next, c_next = self.online_net.step(x_t, h_t, c_t)
            q_np = q_val.squeeze(0).cpu().numpy()
            self.h = h_next.squeeze(0).cpu().numpy()
            self.c = c_next.squeeze(0).cpu().numpy()

        self._last_obs_timestamp = observation.timestamp
        self._last_q_values = np.copy(q_np)
        return np.copy(q_np)

    def get_hidden_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns copies of current hidden state (h, c)."""
        return np.copy(self.h), np.copy(self.c)

    def select_bin(self, observation: SchedulerObservation) -> int:
        """Selects frequency bin via epsilon-greedy in train mode or greedy in eval."""
        q_values = self.get_q_values(observation)

        if self.train_mode and self.rng.random() < self.epsilon:
            bin_idx = int(self.rng.integers(0, self.num_bins))
            reason = f"Torch LSTM-DDQN Epsilon Exploration (eps={self.epsilon:.3f})"
        else:
            bin_idx = int(np.argmax(q_values))
            reason = f"Torch LSTM-DDQN Greedy Exploitation Q={q_values[bin_idx]:.3f}"

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
            "rule": "torch_lstm_ddqn",
        }
        return bin_idx

    def update_policy(
        self,
        observation: SchedulerObservation,
        action: ScanAction | int,
        reward: float,
        next_observation: SchedulerObservation,
        done: bool,
        regime_id: int | None = None,
    ) -> None:
        """Stores transition in replay buffer and executes GPU Double DQN update."""
        if not self.train_mode:
            return

        state = self.encoder.encode(observation)
        next_state = self.encoder.encode(next_observation)
        action_idx = action.frequency_bin if isinstance(action, ScanAction) else int(action)

        h_to_store = np.copy(self._prev_h) if self.use_stored_hidden else None
        c_to_store = np.copy(self._prev_c) if self.use_stored_hidden else None
        r_id = regime_id if regime_id is not None else getattr(self, "current_regime_id", None)

        self.replay_buffer.push(
            state,
            action_idx,
            reward,
            next_state,
            done,
            h=h_to_store,
            c=c_to_store,
            regime_id=r_id,
        )

        if len(self.replay_buffer) >= self.warmup_steps:
            valid_starts = self.replay_buffer.get_valid_start_indices()
            if valid_starts:
                loss = self._train_step()
                self.last_loss = loss
                self.losses.append(loss)

    def _train_step(self) -> float:
        """Samples contiguous sequence minibatch and applies PyTorch Recurrent Double DQN update."""
        self.online_net.train()
        self.target_net.eval()

        sample_out = self.replay_buffer.sample(self.batch_size, return_hidden=True)
        b_states, b_actions, b_rewards, b_next_states, b_dones, b_h0, b_c0 = sample_out

        # Tensors to device
        states_t = torch.from_numpy(b_states).to(self.device)            # [B, L, D]
        actions_t = torch.from_numpy(b_actions).long().to(self.device)     # [B, L]
        rewards_t = torch.from_numpy(b_rewards).to(self.device)           # [B, L]
        next_states_t = torch.from_numpy(b_next_states).to(self.device)   # [B, L, D]
        dones_t = torch.from_numpy(b_dones).to(self.device)               # [B, L]

        if self.use_stored_hidden and b_h0 is not None and b_c0 is not None:
            h0_t = torch.from_numpy(b_h0).to(self.device)  # [B, H]
            c0_t = torch.from_numpy(b_c0).to(self.device)  # [B, H]

            with torch.no_grad():
                # Advance initial hidden state by 1 step for next_states sequence
                _, h_next_online, c_next_online = self.online_net.step(states_t[:, 0, :], h0_t, c0_t)
                _, h_next_target, c_next_target = self.target_net.step(states_t[:, 0, :], h0_t, c0_t)
        else:
            h0_t = None
            c0_t = None
            h_next_online = None
            c_next_online = None
            h_next_target = None
            c_next_target = None

        # 1. Double DQN Next Action Selection (Online Network)
        with torch.no_grad():
            next_q_online, _, _ = self.online_net.forward_sequence(
                next_states_t, h_0=h_next_online, c_0=c_next_online
            )  # [B, L, N]
            best_next_actions = torch.argmax(next_q_online, dim=2, keepdim=True)  # [B, L, 1]

            # 2. Target Network Evaluation at a*
            next_q_target, _, _ = self.target_net.forward_sequence(
                next_states_t, h_0=h_next_target, c_0=c_next_target
            )  # [B, L, N]
            target_next_q = torch.gather(next_q_target, dim=2, index=best_next_actions).squeeze(2)  # [B, L]

            # 3. Bellman Target: y = r + gamma * (1 - done) * Q_target(s', a*)
            target_y = rewards_t + self.gamma * (1.0 - dones_t) * target_next_q  # [B, L]

        # 4. Backpropagation Through Time on Online Network
        self.optimizer.zero_grad()
        q_seq, _, _ = self.online_net.forward_sequence(states_t, h_0=h0_t, c_0=c0_t)  # [B, L, N]
        chosen_q = torch.gather(q_seq, dim=2, index=actions_t.unsqueeze(2)).squeeze(2)  # [B, L]

        if self.burn_in > 0:
            chosen_q = chosen_q[:, self.burn_in :]
            target_y = target_y[:, self.burn_in :]

        td_error = target_y - chosen_q
        loss = torch.mean(td_error ** 2)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.train_step_count += 1

        # 5. Periodic Target Network Copy
        if self.train_step_count % self.target_update_frequency == 0:
            self.target_net.copy_from(self.online_net)

        loss_val = float(loss.item())
        gnorm_val = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        tderr_val = float(torch.mean(torch.abs(td_error)).item())
        qmean_val = float(torch.mean(chosen_q).item())
        qvar_val = float(torch.var(chosen_q).item())

        self.grad_norms.append(gnorm_val)
        self.td_errors.append(tderr_val)
        self.q_means.append(qmean_val)
        self.q_vars.append(qvar_val)

        return loss_val

    def reset(self) -> None:
        """Resets recurrent memory and episode tracking (zero cross-episode leakage)."""
        super().reset()
        self.h = np.zeros(self.hidden_dim, dtype=np.float32)
        self.c = np.zeros(self.hidden_dim, dtype=np.float32)
        self._prev_h = np.zeros(self.hidden_dim, dtype=np.float32)
        self._prev_c = np.zeros(self.hidden_dim, dtype=np.float32)
        self.epsilon = self.epsilon_start if self.train_mode else 0.0
        self._last_obs_timestamp = None
        self._last_q_values = np.zeros(self.num_bins, dtype=np.float32)
        self.replay_buffer.start_new_episode()

    def reseed(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        net_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        buf_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else None
        if net_seed is not None:
            torch.manual_seed(net_seed)
        self.replay_buffer.rng = np.random.default_rng(buf_seed)

    def save_numpy_checkpoint(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Exports model parameters into production NumPy .npz checkpoint format."""
        return self.online_net.save_as_numpy_checkpoint(
            path=path,
            metadata=metadata,
            target_net=self.target_net,
            bands_hz=self.bands_hz,
        )

    def to_numpy_scheduler(self, **kwargs: Any) -> LSTMDDQNScheduler:
        """Instantiates a pure NumPy LSTMDDQNScheduler with identical weights for production inference."""
        numpy_sched = LSTMDDQNScheduler(
            bands_hz=self.bands_hz,
            hidden_dim=self.hidden_dim,
            dense_dim=self.dense_dim,
            sequence_length=self.sequence_length,
            burn_in=self.burn_in,
            use_stored_hidden=self.use_stored_hidden,
            regime_balanced_replay=self.regime_balanced_replay,
            seed=self.seed,
            **kwargs,
        )
        numpy_sched.online_net.load_weights_dict(self.online_net.to_numpy_weights())
        numpy_sched.target_net.load_weights_dict(self.target_net.to_numpy_weights())
        numpy_sched.eval()
        numpy_sched.epsilon = 0.0
        return numpy_sched
