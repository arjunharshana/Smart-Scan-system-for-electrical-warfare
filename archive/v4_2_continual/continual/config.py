from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class V42ContinualConfig:
    """Configuration for V4.2 Continual Adaptive Hybrid Scheduler."""

    # 1. Regime Change Detection
    regime_window: int = 30
    min_shift_evidence_steps: int = 8
    surprise_threshold: float = 0.40
    ddqn_conf_drop_threshold: float = 0.25
    prediction_miss_rate_threshold: float = 0.65
    reward_drop_threshold: float = 0.00
    stablization_evidence_steps: int = 12

    # 2. Online Replay Buffer
    recent_replay_capacity: int = 5000
    historical_replay_capacity: int = 5000
    sequence_length: int = 10
    burn_in: int = 0
    min_replay_sequences: int = 32
    recent_replay_ratio: float = 0.70  # 70% recent, 30% historical

    # 3. Controlled Online Training
    adaptation_updates: int = 15
    adaptation_batch_size: int = 16
    adaptation_learning_rate: float = 0.0005
    adaptation_cooldown_steps: int = 40
    max_updates_per_regime: int = 100
    gradient_clip: float = 5.0
    gamma: float = 0.95

    # 4. Validation Gate & Rollback
    validation_batch_size: int = 16
    max_historical_loss_degradation: float = 1.25  # Max 25% increase in historical loss
    recent_loss_improvement_margin: float = 0.05
    rollback_enabled: bool = True

    # 5. Arbitration Modulation
    ca_weight_floor_suspected: float = 0.85
    ca_weight_floor_rollback: float = 0.80

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_window": self.regime_window,
            "min_shift_evidence_steps": self.min_shift_evidence_steps,
            "surprise_threshold": self.surprise_threshold,
            "ddqn_conf_drop_threshold": self.ddqn_conf_drop_threshold,
            "prediction_miss_rate_threshold": self.prediction_miss_rate_threshold,
            "reward_drop_threshold": self.reward_drop_threshold,
            "stablization_evidence_steps": self.stablization_evidence_steps,
            "recent_replay_capacity": self.recent_replay_capacity,
            "historical_replay_capacity": self.historical_replay_capacity,
            "sequence_length": self.sequence_length,
            "burn_in": self.burn_in,
            "min_replay_sequences": self.min_replay_sequences,
            "recent_replay_ratio": self.recent_replay_ratio,
            "adaptation_updates": self.adaptation_updates,
            "adaptation_batch_size": self.adaptation_batch_size,
            "adaptation_learning_rate": self.adaptation_learning_rate,
            "adaptation_cooldown_steps": self.adaptation_cooldown_steps,
            "max_updates_per_regime": self.max_updates_per_regime,
            "gradient_clip": self.gradient_clip,
            "gamma": self.gamma,
            "validation_batch_size": self.validation_batch_size,
            "max_historical_loss_degradation": self.max_historical_loss_degradation,
            "recent_loss_improvement_margin": self.recent_loss_improvement_margin,
            "rollback_enabled": self.rollback_enabled,
            "ca_weight_floor_suspected": self.ca_weight_floor_suspected,
            "ca_weight_floor_rollback": self.ca_weight_floor_rollback,
        }
