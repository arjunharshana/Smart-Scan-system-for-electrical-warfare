from __future__ import annotations

from abc import ABC, abstractmethod

from rf_environment.domain.emitter import EmitterState
from rf_environment.domain.signal import IdealSignal


class SignalSource(ABC):
    """Optional RF generation backend. GNU Radio can implement this later."""

    @abstractmethod
    def generate(self, emitter_state: EmitterState) -> IdealSignal:
        raise NotImplementedError
