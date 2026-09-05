from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Any
import numpy as np

from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.hybrid.continual.config import V42ContinualConfig


class RegimeState(str, Enum):
    STABLE = "STABLE"
    SUSPECTED_SHIFT = "SUSPECTED_SHIFT"
    ADAPTING = "ADAPTING"
    VALIDATING = "VALIDATING"
    ROLLBACK = "ROLLBACK"


class RegimeChangeDetector:
    """Lightweight, observable-only regime shift detector and state machine.

    Monitors rolling detector and model observables to detect when the predictive
    LSTM-DDQN model is becoming unreliable due to an environmental distribution shift.

    States:
        STABLE: Normal operation, high predictive confidence.
        SUSPECTED_SHIFT: Sustained degradation detected; CA responsibility increased.
        ADAPTING: Sufficient replay collected, candidate training triggered.
        VALIDATING: Candidate model under validation against baseline.
        ROLLBACK: Candidate rejected, previous checkpoint restored.

    Invariants:
    1. Consumes strictly receiver observables (confidence, surprise, misses, reward).
    2. Zero access to ground-truth emitter parameters or state.
    3. Requires persistent evidence (min_shift_evidence_steps) to prevent false alarms on noise.
    """

    def __init__(self, config: V42ContinualConfig | None = None) -> None:
        self.config = config or V42ContinualConfig()
        self.state = RegimeState.STABLE

        W = self.config.regime_window
        self.window_size = W

        # Rolling statistics
        self.q_confidences: deque[float] = deque(maxlen=W)
        self.ca_confidences: deque[float] = deque(maxlen=W)
        self.surprises: deque[float] = deque(maxlen=W)
        self.rewards: deque[float] = deque(maxlen=W)
        self.detection_hits: deque[bool] = deque(maxlen=W)
        self.prediction_misses: deque[bool] = deque(maxlen=W)
        self.transition_consistencies: deque[float] = deque(maxlen=W)

        # State machine counters
        self.consecutive_shift_evidence: int = 0
        self.consecutive_stable_evidence: int = 0
        self.steps_since_last_adaptation: int = 9999
        self.total_detected_shifts: int = 0
        self.total_steps: int = 0

    def update(
        self,
        observation: SchedulerObservation,
        ddqn_confidence: float,
        ca_confidence: float,
        surprise: float,
        online_reward: float,
        transition_consistency: float,
        predicted_bin: int,
        executed_bin: int,
    ) -> RegimeState:
        """Updates rolling statistics and advances the regime state machine."""
        self.total_steps += 1
        self.steps_since_last_adaptation += 1

        detected = bool(observation.last_detection)
        prediction_miss = (predicted_bin == executed_bin) and (not detected)

        # 1. Record rolling observables
        self.q_confidences.append(float(ddqn_confidence))
        self.ca_confidences.append(float(ca_confidence))
        self.surprises.append(float(surprise))
        self.rewards.append(float(online_reward))
        self.detection_hits.append(detected)
        self.prediction_misses.append(prediction_miss)
        self.transition_consistencies.append(float(transition_consistency))

        # 2. Compute window aggregates
        avg_q_conf = float(np.mean(self.q_confidences)) if self.q_confidences else 1.0
        avg_surprise = float(np.mean(self.surprises)) if self.surprises else 0.0
        avg_reward = float(np.mean(self.rewards)) if self.rewards else 0.0
        miss_rate = float(np.mean(self.prediction_misses)) if self.prediction_misses else 0.0
        hit_rate = float(np.mean(self.detection_hits)) if self.detection_hits else 0.0

        # 3. Assess shift evidence at this step
        # Evidence requires multiple concordant signals of predictive degradation
        signals = 0
        if avg_q_conf < self.config.ddqn_conf_drop_threshold:
            signals += 1
        if avg_surprise > self.config.surprise_threshold:
            signals += 1
        if miss_rate > self.config.prediction_miss_rate_threshold:
            signals += 1
        if avg_reward < self.config.reward_drop_threshold:
            signals += 1

        is_shift_evidence = (signals >= 2) and (hit_rate < 0.25)
        is_stable_evidence = (avg_q_conf >= 0.35) and (hit_rate >= 0.30) and (avg_surprise < 0.25)

        if is_shift_evidence:
            self.consecutive_shift_evidence += 1
            self.consecutive_stable_evidence = 0
        else:
            self.consecutive_shift_evidence = max(0, self.consecutive_shift_evidence - 1)

        if is_stable_evidence:
            self.consecutive_stable_evidence += 1
        else:
            self.consecutive_stable_evidence = 0

        # 4. State Machine Transitions
        if self.state == RegimeState.STABLE:
            if self.consecutive_shift_evidence >= self.config.min_shift_evidence_steps:
                self.state = RegimeState.SUSPECTED_SHIFT
                self.total_detected_shifts += 1

        elif self.state == RegimeState.SUSPECTED_SHIFT:
            # If the environment naturally recovered or shift was transient
            if self.consecutive_stable_evidence >= self.config.stablization_evidence_steps:
                self.state = RegimeState.STABLE
                self.consecutive_shift_evidence = 0

        elif self.state == RegimeState.ROLLBACK:
            # Enforce cooldown in rollback state before allowing another adaptation attempt
            if self.steps_since_last_adaptation >= self.config.adaptation_cooldown_steps:
                if self.consecutive_stable_evidence >= self.config.stablization_evidence_steps:
                    self.state = RegimeState.STABLE
                elif is_shift_evidence:
                    self.state = RegimeState.SUSPECTED_SHIFT

        return self.state

    def trigger_adapting(self) -> None:
        """Transitions from SUSPECTED_SHIFT to ADAPTING."""
        self.state = RegimeState.ADAPTING

    def trigger_validating(self) -> None:
        """Transitions from ADAPTING to VALIDATING."""
        self.state = RegimeState.VALIDATING

    def on_validation_accepted(self) -> None:
        """Called when candidate model passes validation."""
        self.state = RegimeState.STABLE
        self.consecutive_shift_evidence = 0
        self.consecutive_stable_evidence = 0
        self.steps_since_last_adaptation = 0

    def on_validation_rejected(self) -> None:
        """Called when candidate model fails validation."""
        self.state = RegimeState.ROLLBACK
        self.consecutive_shift_evidence = 0
        self.steps_since_last_adaptation = 0

    def can_adapt(self, replay_has_enough_data: bool) -> bool:
        """Checks whether online adaptation is allowed given cooldown and buffer state."""
        return (
            self.state == RegimeState.SUSPECTED_SHIFT
            and replay_has_enough_data
            and self.steps_since_last_adaptation >= self.config.adaptation_cooldown_steps
        )

    def get_telemetry(self) -> dict[str, Any]:
        """Returns diagnostic telemetry of regime detection and rolling averages."""
        return {
            "regime_state": self.state.value,
            "consecutive_shift_evidence": self.consecutive_shift_evidence,
            "consecutive_stable_evidence": self.consecutive_stable_evidence,
            "steps_since_adaptation": self.steps_since_last_adaptation,
            "total_detected_shifts": self.total_detected_shifts,
            "rolling_q_conf": float(np.mean(self.q_confidences)) if self.q_confidences else 0.0,
            "rolling_surprise": float(np.mean(self.surprises)) if self.surprises else 0.0,
            "rolling_reward": float(np.mean(self.rewards)) if self.rewards else 0.0,
            "rolling_hit_rate": float(np.mean(self.detection_hits)) if self.detection_hits else 0.0,
            "rolling_miss_rate": float(np.mean(self.prediction_misses)) if self.prediction_misses else 0.0,
        }

    def reset(self) -> None:
        """Resets detector state and clears all rolling histories."""
        self.state = RegimeState.STABLE
        self.q_confidences.clear()
        self.ca_confidences.clear()
        self.surprises.clear()
        self.rewards.clear()
        self.detection_hits.clear()
        self.prediction_misses.clear()
        self.transition_consistencies.clear()
        self.consecutive_shift_evidence = 0
        self.consecutive_stable_evidence = 0
        self.steps_since_last_adaptation = 9999
        self.total_detected_shifts = 0
        self.total_steps = 0
