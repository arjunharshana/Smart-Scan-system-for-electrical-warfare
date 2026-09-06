"""rf_environment/scheduler/belief/config.py
Configuration schema for V5.0 Augmented Belief-State Bayesian Scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BeliefSchedulerConfig:
    """Configuration parameters for V5.0 Bayesian Belief Scheduler."""

    # Sensor likelihood parameters (derived from calibrated detector)
    p_detection: float = 0.95
    p_false_alarm: float = 0.02

    # Hypothesis bank for candidate dwell values
    supported_dwells: tuple[int, ...] = (1, 3, 5)

    # Ablation mode:
    # 'B0' = Frequency belief only (marginalized or D=1)
    # 'B1' = Augmented belief (F, tau, D) with uniform hop transitions
    # 'B2' = Augmented belief (F, tau, D) with online learned hop transition matrix
    ablation_mode: str = "B1"

    # Action selection exploration mode:
    # 'none' = Pure greedy argmax over marginal channel probability
    # 'uncertainty' = Principled coverage bonus based on time since last scan
    exploration_mode: str = "uncertainty"
    uncertainty_weight: float = 0.08
    staleness_threshold: int = 15

    # Online transition estimation smoothing (Laplace/Dirichlet prior)
    transition_smoothing: float = 0.05

    # Numerical stability parameters
    min_probability_floor: float = 1e-12

    # Metadata
    name: str = "v5_belief"

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_detection": self.p_detection,
            "p_false_alarm": self.p_false_alarm,
            "supported_dwells": list(self.supported_dwells),
            "ablation_mode": self.ablation_mode,
            "exploration_mode": self.exploration_mode,
            "uncertainty_weight": self.uncertainty_weight,
            "staleness_threshold": self.staleness_threshold,
            "transition_smoothing": self.transition_smoothing,
            "min_probability_floor": self.min_probability_floor,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeliefSchedulerConfig:
        clean = dict(data)
        if "supported_dwells" in clean and isinstance(clean["supported_dwells"], list):
            clean["supported_dwells"] = tuple(clean["supported_dwells"])
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in clean.items() if k in valid_keys}
        return cls(**filtered)
