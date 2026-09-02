from rf_environment.channel.channel import ChannelModel
from rf_environment.channel.noise import AdditiveNoise
from rf_environment.environment.rf_environment import RFEnvironment
from rf_environment.environment.simulation_clock import SimulationClock
from rf_environment.emitters.radar import RadarEmitter
from rf_environment.frequency_behaviors.fixed import FixedFrequency
from rf_environment.receiver.detector import Detector
from rf_environment.receiver.receiver import Receiver
from rf_environment.scheduler.sequential_scheduler import SequentialScheduler
from rf_environment.time_behaviors.continuous import Continuous


def _env(rx_freq: float, emitter_freq: float = 300e6, pd=1.0, pfa=0.0) -> RFEnvironment:
    emitter = RadarEmitter(
        "E01",
        FixedFrequency(emitter_freq),
        Continuous(),
        power_dbm=-20,
        bandwidth_hz=1e6,
    )
    receiver = Receiver(
        center_frequency_hz=rx_freq,
        instantaneous_bandwidth_hz=10e6,
        detection_threshold_db=0.0,
    )
    return RFEnvironment(
        emitters=[emitter],
        receiver=receiver,
        detector=Detector(detection_threshold_db=0.0, p_detection=pd, p_false_alarm=pfa, seed=1),
        scheduler=SequentialScheduler([rx_freq, 200e6]),
        clock=SimulationClock(total_time_steps=5),
        channel=ChannelModel(noise=AdditiveNoise(noise_floor_dbm=-100, seed=1)),
        spectrum={"min_frequency_hz": 100e6, "max_frequency_hz": 400e6},
    )


def test_missed_scan_when_receiver_off_frequency():
    env = _env(rx_freq=200e6, emitter_freq=300e6, pd=1.0, pfa=0.0)
    result = env.step()
    assert result["ground_truth"]["transmitting_ids"] == ["E01"]
    assert result["observation"]["detected"] is False
    assert result["ground_truth"]["emitters"][0]["frequency_hz"] == 300e6
    assert abs(result["observation"]["receiver_frequency_hz"] - 200e6) < 1.0
    assert "ground_truth" not in result["observation"]


def test_hit_when_receiver_covers_emitter():
    env = _env(rx_freq=300e6, emitter_freq=300e6, pd=1.0, pfa=0.0)
    result = env.step()
    assert result["observation"]["detected"] is True
    assert result["outcome"]["outcome"] == "HIT"


def test_scheduler_observation_omits_full_emitter_truth():
    env = _env(rx_freq=300e6)
    result = env.step()
    obs = result["observation"]
    assert "emitters" not in obs
    assert "transmitting_ids" not in obs
