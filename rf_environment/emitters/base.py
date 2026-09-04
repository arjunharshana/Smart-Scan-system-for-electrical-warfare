from __future__ import annotations

from rf_environment.domain.emitter import EmitterState
from rf_environment.domain.enums import EmitterType
from rf_environment.frequency_behaviors.base import FrequencyBehavior
from rf_environment.time_behaviors.base import TimeBehavior


class BaseEmitter:
    """Determines its own RF state. Contains no receiver logic."""

    emitter_type: EmitterType

    def __init__(
        self,
        emitter_id: str,
        frequency_behavior: FrequencyBehavior,
        time_behavior: TimeBehavior,
        power_dbm: float,
        bandwidth_hz: float,
        modulation: str = "UNKNOWN",
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.emitter_id = emitter_id
        self.frequency_behavior = frequency_behavior
        self.time_behavior = time_behavior
        self.power_dbm = float(power_dbm)
        self.bandwidth_hz = float(bandwidth_hz)
        self.modulation = modulation
        self.position = position
        self._state: EmitterState | None = None

    def step(self, time_step: int) -> EmitterState:
        transmitting = self.time_behavior.is_transmitting(time_step)
        frequency_hz = self.frequency_behavior.get_frequency(time_step, {"transmitting": transmitting})
        dwell_steps = getattr(self.frequency_behavior, "dwell_steps", 1)
        hop_index = time_step // dwell_steps if hasattr(self.frequency_behavior, "dwell_steps") else None
        self._state = EmitterState(
            emitter_id=self.emitter_id,
            emitter_type=self.emitter_type,
            timestamp=time_step,
            transmitting=transmitting,
            frequency_hz=frequency_hz,
            bandwidth_hz=self.bandwidth_hz,
            power_dbm=self.power_dbm,
            frequency_behavior=self.frequency_behavior.behavior_type,
            time_behavior=self.time_behavior.behavior_type,
            modulation=self.modulation,
            position_x=self.position[0],
            position_y=self.position[1],
            position_z=self.position[2],
            dwell_steps=dwell_steps,
            hop_index=hop_index,
        )
        return self._state

    def get_state(self) -> EmitterState:
        if self._state is None:
            raise RuntimeError(f"Emitter {self.emitter_id} has not been stepped yet")
        return self._state

    def get_ground_truth(self) -> EmitterState:
        return self.get_state()
