from __future__ import annotations

from rf_environment.domain.emitter import EmitterState
from rf_environment.domain.signal import IdealSignal
from rf_environment.signal.base import SignalSource


class PythonSignalSource(SignalSource):
    def generate(self, emitter_state: EmitterState) -> IdealSignal:
        return IdealSignal(
            emitter_id=emitter_state.emitter_id,
            timestamp=emitter_state.timestamp,
            frequency_hz=emitter_state.frequency_hz,
            bandwidth_hz=emitter_state.bandwidth_hz,
            power_dbm=emitter_state.power_dbm,
            transmitting=emitter_state.transmitting,
        )
