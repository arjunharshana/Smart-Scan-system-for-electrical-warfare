from __future__ import annotations

import copy
import time
from typing import Any
import numpy as np

from rf_environment.scheduler.rl.lstm_network import LSTMQNetwork


class ModelCheckpointManager:
    """Manages model snapshots, deployment of validated candidates, and rollback on failure.

    Maintains bit-for-bit recoverable checkpoints of:
    1. Online LSTM network parameters and Adam moments.
    2. Target network parameters.
    3. Update versioning and adaptation history telemetry.

    Invariants:
    1. Zero ground truth.
    2. Exact parameter restoration on rollback.
    """

    def __init__(self, production_net: LSTMQNetwork, target_net: LSTMQNetwork) -> None:
        self.production_net = production_net
        self.target_net = target_net

        # Checkpoint storage
        self.checkpoint_online_params: list[np.ndarray] = []
        self.checkpoint_target_params: list[np.ndarray] = []
        self.checkpoint_m_moments: list[np.ndarray] = []
        self.checkpoint_v_moments: list[np.ndarray] = []
        self.checkpoint_adam_t: int = 0
        self.version: int = 0

        # Adaptation audit metrics
        self.adaptation_attempts: int = 0
        self.adaptations_accepted: int = 0
        self.adaptations_rejected: int = 0
        self.rollbacks_performed: int = 0
        self.total_gradient_steps: int = 0
        self.time_spent_adapting_ms: float = 0.0

        # Save initial checkpoint
        self.save_checkpoint()

    def save_checkpoint(self) -> None:
        """Saves current production and target network parameters and Adam state."""
        self.checkpoint_online_params = [np.copy(p) for p in self.production_net.params]
        self.checkpoint_target_params = [np.copy(p) for p in self.target_net.params]
        self.checkpoint_m_moments = [np.copy(m) for m in self.production_net.m_moments]
        self.checkpoint_v_moments = [np.copy(v) for v in self.production_net.v_moments]
        self.checkpoint_adam_t = self.production_net.t
        self.version += 1

    def rollback(self) -> None:
        """Restores production and target networks to the saved checkpoint."""
        # 1. Restore online network parameters and Adam moments
        for target_p, saved_p in zip(self.production_net.params, self.checkpoint_online_params):
            target_p[:] = np.copy(saved_p)
        for target_m, saved_m in zip(self.production_net.m_moments, self.checkpoint_m_moments):
            target_m[:] = np.copy(saved_m)
        for target_v, saved_v in zip(self.production_net.v_moments, self.checkpoint_v_moments):
            target_v[:] = np.copy(saved_v)
        self.production_net.t = self.checkpoint_adam_t

        # 2. Restore target network parameters
        for target_p, saved_p in zip(self.target_net.params, self.checkpoint_target_params):
            target_p[:] = np.copy(saved_p)

        self.rollbacks_performed += 1

    def deploy_candidate(self, candidate_net: LSTMQNetwork) -> None:
        """Deploys validated candidate parameters to the production network and saves checkpoint."""
        # 1. Save checkpoint before overwriting
        self.save_checkpoint()

        # 2. Deploy candidate parameters to production network
        self.production_net.copy_from(candidate_net)
        for m_prod, m_cand in zip(self.production_net.m_moments, candidate_net.m_moments):
            m_prod[:] = np.copy(m_cand)
        for v_prod, v_cand in zip(self.production_net.v_moments, candidate_net.v_moments):
            v_prod[:] = np.copy(v_cand)
        self.production_net.t = candidate_net.t

        # 3. Synchronize target network
        self.target_net.copy_from(self.production_net)

        self.adaptations_accepted += 1

    def record_rejection(self) -> None:
        """Records an adaptation rejection event."""
        self.adaptations_rejected += 1

    def record_attempt(self, gradient_steps: int, duration_ms: float) -> None:
        """Records an adaptation session attempt."""
        self.adaptation_attempts += 1
        self.total_gradient_steps += int(gradient_steps)
        self.time_spent_adapting_ms += float(duration_ms)

    def get_telemetry(self) -> dict[str, Any]:
        """Returns adaptation audit summary."""
        return {
            "version": self.version,
            "adaptation_attempts": self.adaptation_attempts,
            "adaptations_accepted": self.adaptations_accepted,
            "adaptations_rejected": self.adaptations_rejected,
            "rollbacks_performed": self.rollbacks_performed,
            "total_gradient_steps": self.total_gradient_steps,
            "time_spent_adapting_ms": self.time_spent_adapting_ms,
        }

    def reset(self) -> None:
        """Resets adaptation counters while preserving initial checkpoint."""
        self.adaptation_attempts = 0
        self.adaptations_accepted = 0
        self.adaptations_rejected = 0
        self.rollbacks_performed = 0
        self.total_gradient_steps = 0
        self.time_spent_adapting_ms = 0.0
        self.save_checkpoint()
