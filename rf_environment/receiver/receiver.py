from __future__ import annotations

from rf_environment.channel.channel import ChannelOutput
from rf_environment.channel.noise import dbm_to_mw, mw_to_dbm
from rf_environment.domain.receiver import ReceiverMeasurement, ReceiverState
from rf_environment.receiver.tuner import Tuner, bands_overlap


class Receiver:
    def __init__(
        self,
        center_frequency_hz: float,
        instantaneous_bandwidth_hz: float,
        sensitivity_dbm: float = -90.0,
        noise_floor_dbm: float = -100.0,
        detection_threshold_db: float = 6.0,
        tuning_time_ms: float = 0.0,
        time_step_ms: float = 10.0,
    ) -> None:
        self.instantaneous_bandwidth_hz = float(instantaneous_bandwidth_hz)
        self.sensitivity_dbm = float(sensitivity_dbm)
        self.noise_floor_dbm = float(noise_floor_dbm)
        self.detection_threshold_db = float(detection_threshold_db)
        self.tuner = Tuner(time_step_ms=time_step_ms, tuning_time_ms=tuning_time_ms)
        self.tuner.center_frequency_hz = float(center_frequency_hz)

    def tune(self, frequency_hz: float) -> None:
        self.tuner.request_tune(frequency_hz)

    def get_state(self) -> ReceiverState:
        tuning = self.tuner.remaining_steps > 0
        return ReceiverState(
            center_frequency_hz=self.tuner.center_frequency_hz or 0.0,
            instantaneous_bandwidth_hz=self.instantaneous_bandwidth_hz,
            sensitivity_dbm=self.sensitivity_dbm,
            noise_floor_dbm=self.noise_floor_dbm,
            detection_threshold_db=self.detection_threshold_db,
            tuning_time_ms=self.tuner.tuning_time_ms,
            remaining_tune_steps=self.tuner.remaining_steps,
            status="TUNING" if tuning else "SCANNING",
        )

    def observe(self, timestamp: int, channel: ChannelOutput) -> ReceiverMeasurement:
        tuning = self.tuner.tick()
        center = self.tuner.center_frequency_hz or 0.0
        bw = self.instantaneous_bandwidth_hz
        if tuning:
            return ReceiverMeasurement(
                timestamp=timestamp,
                center_frequency_hz=center,
                bandwidth_hz=bw,
                noise_power_dbm=channel.noise_power_dbm,
                tuning=True,
            )
        in_band = []
        powers = []
        for signal in channel.signals:
            if not signal.transmitting:
                continue
            if bands_overlap(center, bw, signal.frequency_hz, signal.bandwidth_hz):
                rx_power = channel.received_power_dbm.get(signal.emitter_id, signal.power_dbm)
                if rx_power >= self.sensitivity_dbm:
                    in_band.append(signal.emitter_id)
                    powers.append(rx_power)
        if powers:
            signal_power = mw_to_dbm(sum(dbm_to_mw(p) for p in powers))
            snr = signal_power - channel.noise_power_dbm
        else:
            signal_power = None
            snr = None
        return ReceiverMeasurement(
            timestamp=timestamp,
            center_frequency_hz=center,
            bandwidth_hz=bw,
            in_band_emitter_ids=in_band,
            signal_power_dbm=signal_power,
            noise_power_dbm=channel.noise_power_dbm,
            snr_db=snr,
            tuning=False,
        )
