from rf_environment.frequency_behaviors.fixed import FixedFrequency
from rf_environment.frequency_behaviors.hopping import FrequencyHopping
from rf_environment.frequency_behaviors.sweep import FrequencySweep
from rf_environment.time_behaviors.burst import Burst
from rf_environment.time_behaviors.continuous import Continuous
from rf_environment.time_behaviors.intermittent import Intermittent
from rf_environment.time_behaviors.periodic import Periodic


def test_fixed_frequency_constant():
    beh = FixedFrequency(100e6)
    assert [beh.get_frequency(t) for t in range(8)] == [100e6] * 8


def test_sequential_hopping():
    beh = FrequencyHopping([100.0, 300.0, 200.0, 400.0], mode="sequential")
    assert [beh.get_frequency(t) for t in range(5)] == [100.0, 300.0, 200.0, 400.0, 100.0]


def test_sweep_increments():
    beh = FrequencySweep(100.0, 200.0, 25.0, wrap_around=True)
    vals = [beh.get_frequency(t) for t in range(5)]
    assert vals == [100.0, 125.0, 150.0, 175.0, 200.0]


def test_continuous_always_on():
    assert all(Continuous().is_transmitting(t) for t in range(8))


def test_periodic_pattern():
    beh = Periodic(on_duration=2, off_duration=2, phase=0)
    assert [int(beh.is_transmitting(t)) for t in range(8)] == [1, 1, 0, 0, 1, 1, 0, 0]


def test_burst_windows():
    beh = Burst(burst_duration=2, interval=5, start_time=0)
    pattern = [int(beh.is_transmitting(t)) for t in range(10)]
    assert pattern == [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]


def test_intermittent_reproducible():
    a = Intermittent(0.3, seed=42)
    b = Intermittent(0.3, seed=42)
    seq_a = [a.is_transmitting(t) for t in range(50)]
    seq_b = [b.is_transmitting(t) for t in range(50)]
    assert seq_a == seq_b
    assert any(seq_a) and not all(seq_a)
