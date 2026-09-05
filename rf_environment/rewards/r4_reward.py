from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rf_environment.domain.action import ScanAction
    from rf_environment.domain.state import SchedulerObservation


class R4RewardCalculator:
    """Production V3 R4 Receiver-Derived Reward Calculator.

    Frozen Formula:
        r_det:
            +1.00 if observation.last_detection is True
             0.00 otherwise

        r_step:
            -0.05 if observation.last_detection is False
             0.00 otherwise

        r_strength:
            if last_detection is True and last_detection_strength is not None:
                q = clip((last_detection_strength + 90.0) / 40.0, 0.0, 1.0)
                r_strength = 0.15 * q
            otherwise:
                r_strength = 0.00

        R4 = r_det + r_step + r_strength

    Expected Numerical Bounds:
        Detection:    R in [1.00, 1.15]
        No Detection: R = -0.05

    Strict Architectural Invariants:
        1. Depends strictly on legitimate scheduler-visible inputs (SchedulerObservation, ScanAction).
        2. Zero imports or access to evaluator, ground-truth, or simulator state.
        3. Never credits retained historical strength on non-detection steps.
        4. Treats noise false alarms (last_detection_strength is None) with r_strength = 0.00.
    """

    def __init__(
        self,
        r_det_value: float = 1.00,
        r_step_penalty: float = -0.05,
        r_strength_max: float = 0.15,
        sensitivity_dbm: float = -90.0,
        dynamic_range_db: float = 40.0,
    ) -> None:
        self.r_det_value = float(r_det_value)
        self.r_step_penalty = float(r_step_penalty)
        self.r_strength_max = float(r_strength_max)
        self.sensitivity_dbm = float(sensitivity_dbm)
        self.dynamic_range_db = float(dynamic_range_db)

    def compute(
        self,
        observation: SchedulerObservation | None = None,
        action: ScanAction | None = None,
        *,
        last_detection: bool | None = None,
        last_detection_strength: float | None = None,
    ) -> float:
        """Computes the R4 reward from observation observables."""
        if observation is not None:
            det = bool(observation.last_detection)
            strength = observation.last_detection_strength
        else:
            det = bool(last_detection) if last_detection is not None else False
            strength = last_detection_strength

        # 1. Detection term
        r_det = self.r_det_value if det else 0.00

        # 2. Step search penalty
        r_step = 0.00 if det else self.r_step_penalty

        # 3. Signal strength quality bonus (only credited on positive detection)
        if det and strength is not None:
            q = min(max((float(strength) - self.sensitivity_dbm) / self.dynamic_range_db, 0.0), 1.0)
            r_strength = self.r_strength_max * q
        else:
            r_strength = 0.00

        return float(r_det + r_step + r_strength)
