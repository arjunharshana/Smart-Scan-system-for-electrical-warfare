from __future__ import annotations

import time
from typing import Any
import numpy as np

from rf_environment.scheduler.hybrid.continual.config import V42ContinualConfig
from rf_environment.scheduler.hybrid.continual.online_replay import OnlineReplayBuffer
from rf_environment.scheduler.rl.lstm_network import LSTMQNetwork


class OnlineTrainer:
    """Controlled online trainer for candidate LSTM-DDQN models.

    Executes a bounded number of Backpropagation Through Time (BPTT) parameter
    updates on an isolated candidate network copy using mixed experience replay.

    Invariants:
    1. Production network weights are NEVER modified directly during training.
    2. Updates are bounded by config.adaptation_updates (no runaway loops).
    3. Strictly zero ground truth.
    """

    def __init__(self, config: V42ContinualConfig | None = None) -> None:
        self.config = config or V42ContinualConfig()

    def train_candidate(
        self,
        production_net: LSTMQNetwork,
        target_net: LSTMQNetwork,
        replay_buffer: OnlineReplayBuffer,
    ) -> tuple[LSTMQNetwork, dict[str, Any]]:
        """Clones production model and trains candidate on mixed replay sequences."""
        t_start = time.perf_counter()

        # 1. Clone candidate network from production model
        candidate_net = production_net.clone()
        # Set learning rate and clipping for online adaptation
        candidate_net.lr = self.config.adaptation_learning_rate
        candidate_net.grad_clip = self.config.gradient_clip

        # Copy Adam moments to continue optimization smoothly
        for m_cand, m_prod in zip(candidate_net.m_moments, production_net.m_moments):
            m_cand[:] = np.copy(m_prod)
        for v_cand, v_prod in zip(candidate_net.v_moments, production_net.v_moments):
            v_cand[:] = np.copy(v_prod)
        candidate_net.t = production_net.t

        losses: list[float] = []
        batch_size = self.config.adaptation_batch_size
        gamma = self.config.gamma

        # 2. Execute bounded BPTT gradient updates
        for step in range(self.config.adaptation_updates):
            # Sample mixed batch (70% recent / 30% historical)
            b_states, b_actions, b_rewards, b_next_states, b_dones = replay_buffer.sample_mixed(batch_size)

            # Double DQN Next Action Selection (using Candidate Network)
            next_q_online, _, _ = candidate_net.forward_sequence(b_next_states)
            best_next_actions = np.argmax(next_q_online, axis=2)

            # Target Network Evaluation at candidate's greedy actions
            next_q_target, _, _ = target_net.forward_sequence(b_next_states)
            target_next_q = np.take_along_axis(
                next_q_target, best_next_actions[:, :, np.newaxis], axis=2
            ).squeeze(2)

            # Bellman targets: y = r + gamma * (1 - done) * Q_target(s', a*)
            targets = b_rewards + gamma * (1.0 - b_dones) * target_next_q

            # BPTT gradient update on candidate network
            loss = candidate_net.train_step(
                b_states, b_actions, targets, burn_in=self.config.burn_in
            )
            losses.append(float(loss))

        duration_ms = (time.perf_counter() - t_start) * 1000.0

        stats = {
            "gradient_steps": len(losses),
            "initial_loss": losses[0] if losses else 0.0,
            "final_loss": losses[-1] if losses else 0.0,
            "duration_ms": duration_ms,
        }

        return candidate_net, stats
