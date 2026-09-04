from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rf_environment.domain.action import ScanAction
from rf_environment.domain.enums import Serializable
from rf_environment.domain.state import SchedulerObservation


class PredictionRecord(Serializable):
    """Record of an observed transition frequency prediction."""

    step: int
    previous_frequency_hz: float
    predicted_frequency_hz: float
    actual_observed_frequency_hz: float | None = None
    correct: bool | None = None
    confidence: float = 0.0


class ObservedTransitionStats(Serializable):
    """Aggregated statistics of observed receiver detections."""

    frequencies_hz: list[float] = []
    counts: dict[str, dict[str, int]] = {}
    probabilities: dict[str, dict[str, float]] = {}
    total_transitions: int = 0
    total_predictions: int = 0
    correct_predictions: int = 0
    prediction_accuracy: float = 0.0


@dataclass(frozen=True)
class Transition:
    """Canonical RL transition tuple (o_t, a_t, r_t, o_{t+1}, done)."""

    observation: SchedulerObservation
    action: ScanAction
    reward: float
    next_observation: SchedulerObservation
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Complete sequence of transitions spanning one simulation episode."""

    episode_id: int
    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    total_steps: int = 0

    def append(self, transition: Transition) -> None:
        self.transitions.append(transition)
        self.total_reward += transition.reward
        self.total_steps += 1


@dataclass
class StepResult:
    """Hybrid step result supporting both standard RL 4-tuple unpacking and legacy dictionary access.

    Examples:
        # Standard RL Gym unpack:
        next_obs, reward, done, info = env.step(action)

        # Legacy dict access:
        gt = result["ground_truth"]
        obs = result["observation"]
        metrics = result["metrics"]
    """

    observation: SchedulerObservation
    reward: float
    done: bool
    info: dict[str, Any]

    def __iter__(self):
        return iter((self.observation, self.reward, self.done, self.info))

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return (self.observation, self.reward, self.done, self.info)[key]
        return self.info[key]

    def __contains__(self, key: str) -> bool:
        return key in self.info

    def get(self, key: str, default: Any = None) -> Any:
        return self.info.get(key, default)
