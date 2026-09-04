from __future__ import annotations

from typing import Any
from rf_environment.domain.enums import Serializable


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
    # Source freq str (MHz) -> target freq str (MHz) -> count
    counts: dict[str, dict[str, int]] = {}
    # Source freq str (MHz) -> target freq str (MHz) -> probability
    probabilities: dict[str, dict[str, float]] = {}
    total_transitions: int = 0
    total_predictions: int = 0
    correct_predictions: int = 0
    prediction_accuracy: float = 0.0
