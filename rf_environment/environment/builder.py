from __future__ import annotations

from typing import Any

from rf_environment.channel.channel import ChannelModel
from rf_environment.channel.noise import AdditiveNoise
from rf_environment.channel.propagation import FreeSpacePathLoss
from rf_environment.environment.rf_environment import RFEnvironment
from rf_environment.environment.scenario import emitters_from_scenario, load_scenario
from rf_environment.environment.simulation_clock import SimulationClock
from rf_environment.receiver.detector import Detector
from rf_environment.receiver.receiver import Receiver
from rf_environment.scheduler.factory import create_scheduler, default_scan_bands
from rf_environment.util.seeding import derive_seed


def build_environment(
    scenario: dict[str, Any],
    scheduler_name: str | None = None,
    **scheduler_kwargs: Any,
) -> RFEnvironment:
    sim = scenario.get("simulation", {})
    seed = int(sim.get("seed", 0))
    spectrum = scenario.get("spectrum", {})
    min_hz = float(spectrum.get("min_frequency_hz", 100_000_000))
    max_hz = float(spectrum.get("max_frequency_hz", 1_000_000_000))
    rx_cfg = scenario.get("receiver", {})
    bw = float(rx_cfg.get("instantaneous_bandwidth_hz", 10_000_000))
    time_step_ms = float(sim.get("time_step_ms", 10))
    clock = SimulationClock(
        time_step_ms=time_step_ms,
        total_time_steps=sim.get("total_time_steps"),
    )
    receiver = Receiver(
        center_frequency_hz=float(rx_cfg.get("center_frequency_hz", min_hz + bw / 2)),
        instantaneous_bandwidth_hz=bw,
        sensitivity_dbm=float(rx_cfg.get("sensitivity_dbm", -90)),
        noise_floor_dbm=float(rx_cfg.get("noise_floor_dbm", -100)),
        detection_threshold_db=float(rx_cfg.get("detection_threshold_db", 6)),
        tuning_time_ms=float(rx_cfg.get("tuning_time_ms", 0)),
        time_step_ms=time_step_ms,
    )
    det_cfg = scenario.get("detector", {})
    detector = Detector(
        detection_threshold_db=float(det_cfg.get("detection_threshold_db", rx_cfg.get("detection_threshold_db", 6))),
        p_detection=float(det_cfg.get("p_detection", 0.9)),
        p_false_alarm=float(det_cfg.get("p_false_alarm", 0.02)),
        seed=derive_seed(seed, "detector"),
    )
    sched_cfg = dict(scenario.get("scheduler", {}))
    sched_cfg.update(scheduler_kwargs)
    name = scheduler_name or sched_cfg.get("type", "sequential")
    bands = sched_cfg.get("bands_hz") or default_scan_bands(min_hz, max_hz, bw)
    scheduler = create_scheduler(name, bands, seed=derive_seed(seed, "scheduler"), config=sched_cfg)
    channel_cfg = scenario.get("channel", {})
    channel = ChannelModel(
        noise=AdditiveNoise(
            noise_floor_dbm=float(channel_cfg.get("noise_floor_dbm", rx_cfg.get("noise_floor_dbm", -100))),
            seed=derive_seed(seed, "channel"),
        ),
        path_loss=FreeSpacePathLoss(enabled=bool(channel_cfg.get("path_loss", False))),
    )
    emitters = emitters_from_scenario(scenario)
    return RFEnvironment(
        emitters=emitters,
        receiver=receiver,
        detector=detector,
        scheduler=scheduler,
        clock=clock,
        channel=channel,
        spectrum={"min_frequency_hz": min_hz, "max_frequency_hz": max_hz},
    )


def build_from_path(path: str, scheduler_name: str | None = None, **scheduler_kwargs: Any) -> RFEnvironment:
    return build_environment(load_scenario(path), scheduler_name=scheduler_name, **scheduler_kwargs)
