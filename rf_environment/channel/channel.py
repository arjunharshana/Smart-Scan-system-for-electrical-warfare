from __future__ import annotations

from rf_environment.channel.interference import Interference
from rf_environment.channel.noise import AdditiveNoise
from rf_environment.channel.propagation import FreeSpacePathLoss
from rf_environment.domain.emitter import EmitterState
from rf_environment.domain.signal import IdealSignal


class ChannelOutput:
    def __init__(
        self,
        signals: list[IdealSignal],
        received_power_dbm: dict[str, float],
        noise_power_dbm: float,
    ) -> None:
        self.signals = signals
        self.received_power_dbm = received_power_dbm
        self.noise_power_dbm = noise_power_dbm


class ChannelModel:
    def __init__(
        self,
        noise: AdditiveNoise | None = None,
        path_loss: FreeSpacePathLoss | None = None,
        interference: Interference | None = None,
        rx_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.noise = noise or AdditiveNoise()
        self.path_loss = path_loss or FreeSpacePathLoss(enabled=False)
        self.interference = interference or Interference()
        self.rx_position = rx_position

    def apply(self, signals: list[IdealSignal], emitter_states: list[EmitterState]) -> ChannelOutput:
        by_id = {e.emitter_id: e for e in emitter_states}
        received: dict[str, float] = {}
        for signal in signals:
            state = by_id.get(signal.emitter_id)
            atten = self.path_loss.attenuation_db(state, self.rx_position) if state else 0.0
            received[signal.emitter_id] = signal.power_dbm - atten
        noise = self.noise.sample_dbm()
        return ChannelOutput(signals=signals, received_power_dbm=received, noise_power_dbm=noise)
