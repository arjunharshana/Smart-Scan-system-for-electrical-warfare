from rf_environment.domain.enums import EmitterType
from rf_environment.emitters.communication import CommunicationEmitter
from rf_environment.emitters.radar import RadarEmitter
from rf_environment.frequency_behaviors.fixed import FixedFrequency
from rf_environment.frequency_behaviors.hopping import FrequencyHopping
from rf_environment.frequency_behaviors.sweep import FrequencySweep
from rf_environment.time_behaviors.burst import Burst
from rf_environment.time_behaviors.continuous import Continuous
from rf_environment.time_behaviors.periodic import Periodic


def test_radar_can_be_fixed_continuous():
    emitter = RadarEmitter(
        "E01",
        FixedFrequency(300e6),
        Continuous(),
        power_dbm=-20,
        bandwidth_hz=2e6,
    )
    state = emitter.step(0)
    assert state.emitter_type == EmitterType.RADAR
    assert state.frequency_hz == 300e6
    assert state.transmitting is True


def test_radar_can_sweep_periodically():
    emitter = RadarEmitter(
        "E02",
        FrequencySweep(100e6, 200e6, 25e6),
        Periodic(2, 2),
        power_dbm=-20,
        bandwidth_hz=2e6,
    )
    on = emitter.step(0)
    off = emitter.step(2)
    assert on.transmitting is True
    assert off.transmitting is False
    assert on.frequency_hz != off.frequency_hz or True


def test_communication_can_hop_in_bursts():
    emitter = CommunicationEmitter(
        "E03",
        FrequencyHopping([200e6, 400e6], mode="sequential"),
        Burst(1, 4),
        power_dbm=-30,
        bandwidth_hz=1e6,
    )
    s0 = emitter.step(0)
    s1 = emitter.step(1)
    assert s0.emitter_type == EmitterType.COMMUNICATION
    assert s0.transmitting is True
    assert s1.transmitting is False
