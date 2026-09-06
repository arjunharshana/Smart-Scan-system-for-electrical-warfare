from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

WhittleAblationMode = Literal["W0", "W1", "W2", "W3"]


@dataclass
class WhittleConfig:
    """Configuration parameters for the Whittle-Style Heuristic Index Scheduler.

    Ablation Modes:
    - W0: Belief only (Index = V_belief)
    - W1: Belief + Recency (Index = V_belief + mu * V_recency)
    - W2: Belief + Recency + Uncertainty (Index = V_belief + mu * V_recency + lambda * V_unc)
    - W3: Full Model (Belief + Recency + Uncertainty + Dwell Persistence + Periodicity)
    """

    ablation_mode: WhittleAblationMode = "W3"

    # Index Term Weights
    w_belief: float = 1.0
    w_recency: float = 0.35       # mu: Weight on empirical transition likelihood
    w_uncertainty: float = 0.20   # lambda: Weight on information / coverage incentive
    w_dwell: float = 0.40         # eta: Weight on remaining dwell duration persistence
    w_periodic: float = 0.25      # zeta: Weight on estimated return period synchronization

    # Passive Belief & Uncertainty Dynamics
    belief_decay: float = 0.95    # Passive Markovian memory retention rate
    stale_threshold: int = 20     # Steps until uncertainty saturates to 1.0
    background_prior: float | None = None


    # Observation Likelihoods (Receiver/Detector characteristics)
    p_detection: float = 0.95
    p_false_alarm: float = 0.02

    # Temporal Dynamics Parameters
    default_dwell: int = 3        # Baseline estimated dwell duration
    sigma_periodic: float = 2.0   # Bandwidth for Gaussian return interval matching
    transition_smoothing: float = 0.05

    # Decision Rule
    tie_breaker: str = "min_index"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_mode": self.ablation_mode,
            "w_belief": self.w_belief,
            "w_recency": self.w_recency,
            "w_uncertainty": self.w_uncertainty,
            "w_dwell": self.w_dwell,
            "w_periodic": self.w_periodic,
            "belief_decay": self.belief_decay,
            "stale_threshold": self.stale_threshold,
            "background_prior": self.background_prior,
            "p_detection": self.p_detection,
            "p_false_alarm": self.p_false_alarm,
            "default_dwell": self.default_dwell,
            "sigma_periodic": self.sigma_periodic,
            "transition_smoothing": self.transition_smoothing,
            "tie_breaker": self.tie_breaker,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WhittleConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
