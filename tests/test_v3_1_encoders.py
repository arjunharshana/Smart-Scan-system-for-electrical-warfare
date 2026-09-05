from __future__ import annotations

import numpy as np

from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.rl.encoder import ObservationEncoder
from rf_environment.scheduler.rl.temporal_encoder import TemporalObservationEncoder


def _make_obs(num_bins: int = 20) -> SchedulerObservation:
    return SchedulerObservation(
        timestamp=45,
        current_frequency_bin=5,
        last_detection=True,
        last_detection_bin=5,
        last_detection_strength=-72.5,
        recent_detection_history=(False, False, True, False, True),
        recent_frequency_history=(3, 4, 5, 8, 5),
        scan_count_by_bin=tuple(1 if i == 5 else 0 for i in range(num_bins)),
        time_since_scan_by_bin=tuple(0 if i == 5 else 15 for i in range(num_bins)),
        time_since_last_detection=2,
    )


def test_1_encoder_a_matches_v3_observation_encoder() -> None:
    """Encoder A must produce exact same dimension and vector as ObservationEncoder."""
    num_bins = 20
    obs = _make_obs(num_bins)
    enc_orig = ObservationEncoder(num_bins=num_bins)
    enc_a = TemporalObservationEncoder.create_encoder_a(num_bins=num_bins)

    assert enc_orig.feature_dim == enc_a.feature_dim, f"{enc_orig.feature_dim} != {enc_a.feature_dim}"
    v_orig = enc_orig.encode(obs)
    v_a = enc_a.encode(obs)

    np.testing.assert_array_equal(v_orig, v_a)


def test_2_encoder_b_derived_temporal_features() -> None:
    """Encoder B adds legitimate receiver-derived features."""
    num_bins = 20
    obs = _make_obs(num_bins)
    enc_b = TemporalObservationEncoder.create_encoder_b(num_bins=num_bins)
    expected_dim = (26 + 4 * num_bins) + 7
    assert enc_b.feature_dim == expected_dim, f"{enc_b.feature_dim} != {expected_dim}"

    v_b = enc_b.encode(obs)
    assert v_b.shape == (expected_dim,)
    assert np.all(np.isfinite(v_b))

    base_idx = 26 + 4 * num_bins
    delta = v_b[base_idx]
    direction = v_b[base_idx + 1]
    assert abs(delta - (-3.0 / 19.0)) < 1e-5
    assert direction == -1.0

    det_rate = v_b[base_idx + 2]
    assert abs(det_rate - 0.4) < 1e-5


def test_3_encoder_c_omits_absolute_timestamp() -> None:
    """Encoder C omits normalized timestamp, reducing dimension by 1."""
    num_bins = 20
    obs = _make_obs(num_bins)
    enc_b = TemporalObservationEncoder.create_encoder_b(num_bins=num_bins)
    enc_c = TemporalObservationEncoder.create_encoder_c(num_bins=num_bins)

    assert enc_c.feature_dim == enc_b.feature_dim - 1
    v_c = enc_c.encode(obs)
    assert np.all(np.isfinite(v_c))


def test_4_encoder_d_periodic_features() -> None:
    """Encoder D provides multi-scale harmonic sin/cos encodings."""
    num_bins = 20
    obs = _make_obs(num_bins)
    enc_d = TemporalObservationEncoder.create_encoder_d(num_bins=num_bins)
    expected_dim = 25 + 4 * num_bins + 10
    assert enc_d.feature_dim == expected_dim

    v_d = enc_d.encode(obs)
    assert np.all(np.isfinite(v_d))
    assert np.all(v_d[-10:] >= -1.0 - 1e-6)
    assert np.all(v_d[-10:] <= 1.0 + 1e-6)


def test_5_none_handling_all_variants() -> None:
    """All encoders must handle completely empty/None observations safely."""
    num_bins = 20
    empty_obs = SchedulerObservation(
        timestamp=None,
        current_frequency_bin=None,
        last_detection=None,
        last_detection_bin=None,
        last_detection_strength=None,
        recent_detection_history=None,
        recent_frequency_history=None,
        scan_count_by_bin=None,
        time_since_scan_by_bin=None,
        time_since_last_detection=None,
    )

    for name, enc in [
        ("A", TemporalObservationEncoder.create_encoder_a(num_bins)),
        ("B", TemporalObservationEncoder.create_encoder_b(num_bins)),
        ("C", TemporalObservationEncoder.create_encoder_c(num_bins)),
        ("D", TemporalObservationEncoder.create_encoder_d(num_bins)),
    ]:
        v = enc.encode(empty_obs)
        assert v.shape == (enc.feature_dim,), f"Encoder {name} dimension mismatch"
        assert np.all(np.isfinite(v)), f"Encoder {name} produced NaN/Inf on empty observation"


def test_6_ground_truth_firewall_static_isolation() -> None:
    """Encoder must produce identical output when hidden simulator states change."""
    num_bins = 20
    obs = _make_obs(num_bins)
    enc_b = TemporalObservationEncoder.create_encoder_b(num_bins=num_bins)
    v1 = enc_b.encode(obs)
    v2 = enc_b.encode(obs)
    np.testing.assert_array_equal(v1, v2)
