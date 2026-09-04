from __future__ import annotations

from typing import Sequence
from rf_environment.domain.action import ScanAction, validate_action
from rf_environment.receiver.receiver import Receiver


class ScanController:
    """Translates scheduler ScanAction(frequency_bin) into receiver tuning commands.

    The scheduler selects only the discrete frequency bin.
    Physical receiver frequency calculation, spectrum limits, and tuning are owned here.
    """

    def __init__(
        self,
        receiver: Receiver,
        bands_hz: Sequence[float],
        min_freq_hz: float | None = None,
        max_freq_hz: float | None = None,
    ) -> None:
        self.receiver = receiver
        self.bands_hz = [float(b) for b in bands_hz]
        self.min_freq_hz = min_freq_hz
        self.max_freq_hz = max_freq_hz
        self.current_action: ScanAction | None = None
        self._current_bin: int = 0

    @property
    def current_frequency_hz(self) -> float:
        return self.receiver.center_frequency_hz

    @property
    def current_frequency_bin(self) -> int:
        return self._current_bin

    def validate(self, action: ScanAction) -> None:
        validate_action(action=action, num_bins=len(self.bands_hz))

    def execute_action(self, action: ScanAction) -> float:
        """Validates action and tunes receiver to the physical frequency of the selected bin."""
        self.validate(action)
        self.current_action = action
        self._current_bin = action.frequency_bin
        target_freq = self.bands_hz[action.frequency_bin]
        self.receiver.tune(target_freq)
        return target_freq

    def reset(self, initial_bin: int = 0) -> None:
        self.current_action = None
        self._current_bin = initial_bin if 0 <= initial_bin < len(self.bands_hz) else 0
        init_freq = self.bands_hz[self._current_bin] if self.bands_hz else self.receiver.center_frequency_hz
        self.receiver.tune(init_freq)
