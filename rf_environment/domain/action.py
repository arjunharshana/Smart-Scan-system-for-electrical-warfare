from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanAction:
    """Canonical V3 Scan Action.

    The scheduler commands WHICH frequency bin to scan:
        frequency_bin in {0, 1, ..., N-1}
    Physical frequency tuning and RF parameters are owned by the environment/scan controller.
    """

    frequency_bin: int


def validate_action(action: ScanAction, num_bins: int) -> None:
    """Validates that a ScanAction is within legal bounds.

    Raises:
        TypeError: If action is not a ScanAction instance.
        ValueError: If frequency_bin is out of range [0, num_bins - 1].
    """
    if not isinstance(action, ScanAction):
        raise TypeError(f"Action must be a ScanAction instance, got {type(action).__name__}")

    if not (0 <= action.frequency_bin < num_bins):
        raise ValueError(
            f"Invalid frequency_bin {action.frequency_bin}. Must be in [0, {num_bins - 1}]."
        )
