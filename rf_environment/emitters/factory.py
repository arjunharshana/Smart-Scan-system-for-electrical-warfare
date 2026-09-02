from __future__ import annotations

from typing import Any

from rf_environment.domain.enums import EmitterType
from rf_environment.emitters.base import BaseEmitter
from rf_environment.emitters.communication import CommunicationEmitter
from rf_environment.emitters.radar import RadarEmitter
from rf_environment.frequency_behaviors.fixed import FixedFrequency
from rf_environment.frequency_behaviors.hopping import FrequencyHopping
from rf_environment.frequency_behaviors.random import RandomFrequency
from rf_environment.frequency_behaviors.sweep import FrequencySweep
from rf_environment.time_behaviors.burst import Burst
from rf_environment.time_behaviors.continuous import Continuous
from rf_environment.time_behaviors.intermittent import Intermittent
from rf_environment.time_behaviors.periodic import Periodic
from rf_environment.util.seeding import derive_seed

_EMITTER_TYPES: dict[str, type[BaseEmitter]] = {
    "radar": RadarEmitter,
    "communication": CommunicationEmitter,
    EmitterType.RADAR.value: RadarEmitter,
    EmitterType.COMMUNICATION.value: CommunicationEmitter,
}


def build_frequency_behavior(config: dict[str, Any], seed: int, emitter_id: str):
    kind = str(config["type"]).lower()
    derived = derive_seed(seed, emitter_id, "frequency")
    if kind == "fixed":
        return FixedFrequency(config["frequency_hz"])
    if kind in {"hop", "hopping"}:
        return FrequencyHopping(
            frequencies_hz=config["frequencies_hz"],
            mode=config.get("mode", "sequential"),
            dwell_steps=config.get("dwell_steps", 1),
            seed=derived,
            sequence=config.get("sequence"),
        )
    if kind == "sweep":
        return FrequencySweep(
            start_hz=config["start_hz"] if "start_hz" in config else config["start_frequency_hz"],
            end_hz=config["end_hz"] if "end_hz" in config else config["end_frequency_hz"],
            step_hz=config["step_hz"] if "step_hz" in config else config.get("step_size_hz", 1_000_000),
            direction=config.get("direction", "up"),
            wrap_around=config.get("wrap_around", True),
            dwell_steps=config.get("dwell_steps", 1),
        )
    if kind == "random":
        return RandomFrequency(
            frequencies_hz=config.get("frequencies_hz"),
            min_frequency_hz=config.get("min_frequency_hz"),
            max_frequency_hz=config.get("max_frequency_hz"),
            quantization_hz=config.get("quantization_hz", 1_000_000),
            seed=derived,
            dwell_steps=config.get("dwell_steps", 1),
        )
    raise ValueError(f"Unknown frequency behavior: {kind}")


def build_time_behavior(config: dict[str, Any], seed: int, emitter_id: str):
    kind = str(config["type"]).lower()
    derived = derive_seed(seed, emitter_id, "time")
    if kind == "continuous":
        return Continuous()
    if kind == "periodic":
        return Periodic(
            on_duration=config["on_duration"],
            off_duration=config["off_duration"],
            phase=config.get("phase", 0),
        )
    if kind == "burst":
        return Burst(
            burst_duration=config["burst_duration"],
            interval=config["interval"],
            start_time=config.get("start_time", 0),
        )
    if kind in {"intermittent", "stochastic", "random"}:
        return Intermittent(p_transmit=config.get("p_transmit", 0.3), seed=derived)
    raise ValueError(f"Unknown time behavior: {kind}")


def create_emitter(config: dict[str, Any], seed: int = 0) -> BaseEmitter:
    emitter_id = str(config["id"])
    type_key = str(config["type"]).lower()
    cls = _EMITTER_TYPES.get(type_key)
    if cls is None:
        raise ValueError(f"Unknown emitter type: {config['type']}")
    position = config.get("position", [0.0, 0.0, 0.0])
    return cls(
        emitter_id=emitter_id,
        frequency_behavior=build_frequency_behavior(config["frequency_behavior"], seed, emitter_id),
        time_behavior=build_time_behavior(config["time_behavior"], seed, emitter_id),
        power_dbm=config.get("power_dbm", -20.0),
        bandwidth_hz=config.get("bandwidth_hz", 1_000_000.0),
        modulation=config.get("modulation", "UNKNOWN"),
        position=tuple(position),  # type: ignore[arg-type]
    )
