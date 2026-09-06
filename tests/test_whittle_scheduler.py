from __future__ import annotations

import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.factory import create_scheduler
from rf_environment.scheduler.whittle.belief_state import BanditBeliefTracker
from rf_environment.scheduler.whittle.config import WhittleConfig
from rf_environment.scheduler.whittle.index_policy import WhittleIndexPolicy
from rf_environment.scheduler.whittle.scheduler import WhittleScheduler
from rf_environment.scheduler.whittle.transition_estimator import ObservableTransitionEstimator


def _make_dummy_obs(
    bin_idx: int = 0,
    last_detection: bool = False,
    timestamp: float = 1.0,
    num_bins: int = 10,
) -> SchedulerObservation:
    return SchedulerObservation(
        timestamp=timestamp,
        current_frequency_bin=bin_idx,
        last_detection=last_detection,
        last_detection_bin=bin_idx if last_detection else None,
        last_detection_strength=-65.0 if last_detection else None,
        recent_detection_history=tuple([last_detection] * 5),
        recent_frequency_history=tuple([bin_idx] * 5),
        scan_count_by_bin=tuple([1] * num_bins),
        time_since_scan_by_bin=tuple([1.0] * num_bins),
        time_since_last_detection=0.0 if last_detection else 1.0,
    )


def test_initial_belief_state() -> None:
    """Verifies that all arms initialize with uniform clean priors and maximum uncertainty."""
    tracker = BanditBeliefTracker(num_bins=10)
    beliefs = tracker.get_beliefs()
    uncs = tracker.get_uncertainties()

    assert len(beliefs) == 10
    assert np.allclose(beliefs, tracker.prior)
    assert np.all(uncs == 1.0)
    assert tracker.last_detected_bin is None


def test_belief_update_after_detection() -> None:
    """Verifies Bayesian belief increase and uncertainty collapse on detection."""
    tracker = BanditBeliefTracker(num_bins=10)
    prior_0 = tracker.arms[0].belief

    tracker.update_observation(scanned_bin=0, detection=True, timestamp=1.0)

    assert tracker.arms[0].belief > prior_0
    assert tracker.arms[0].belief > 0.80
    assert tracker.arms[0].uncertainty == 0.0
    assert tracker.arms[0].consecutive_detections == 1
    assert tracker.arms[0].time_since_detection == 0
    assert tracker.last_detected_bin == 0


def test_belief_update_after_miss() -> None:
    """Verifies Bayesian belief drop and zero streak on miss."""
    tracker = BanditBeliefTracker(num_bins=10)
    # First hit
    tracker.update_observation(scanned_bin=0, detection=True, timestamp=1.0)
    hit_belief = tracker.arms[0].belief
    assert tracker.arms[0].consecutive_detections == 1

    # First miss drops belief sharply from hit_belief
    tracker.update_observation(scanned_bin=0, detection=False, timestamp=2.0)
    assert tracker.arms[0].belief < hit_belief
    assert tracker.arms[0].belief < 0.30
    assert tracker.arms[0].uncertainty == 0.0
    assert tracker.arms[0].consecutive_detections == 0
    assert tracker.arms[0].time_since_detection == 1

    # Second consecutive miss drops belief near zero
    tracker.update_observation(scanned_bin=0, detection=False, timestamp=3.0)
    assert tracker.arms[0].belief < 0.05


def test_uncertainty_aging_on_passive_arms() -> None:
    """Verifies that unobserved arms age in uncertainty and decay in belief."""
    config = WhittleConfig(stale_threshold=10, belief_decay=0.90)
    tracker = BanditBeliefTracker(num_bins=10, config=config)

    # Initial scan of arm 1 resets its timer to 0
    tracker.update_observation(scanned_bin=1, detection=True, timestamp=0.0)
    tracker.arms[1].belief = 0.80

    # Scan arm 0 repeatedly for 5 steps
    for t in range(1, 6):
        tracker.update_observation(scanned_bin=0, detection=False, timestamp=float(t))

    # Arm 1 was passive for 5 steps
    assert tracker.arms[1].time_since_scan == 5
    assert tracker.arms[1].uncertainty == 5.0 / 10.0  # 0.5
    assert tracker.arms[1].belief < 0.80             # Decayed


def test_index_calculation_ablation_modes() -> None:
    """Verifies that W0, W1, W2, W3 index calculation terms activate properly."""
    for mode in ["W0", "W1", "W2", "W3"]:
        cfg = WhittleConfig(ablation_mode=mode)
        tracker = BanditBeliefTracker(num_bins=5, config=cfg)
        trans = ObservableTransitionEstimator(num_bins=5)
        policy = WhittleIndexPolicy(config=cfg)

        indices, details = policy.compute_indices(tracker, trans)
        assert len(indices) == 5
        assert not np.any(np.isnan(indices))
        assert details["ablation_mode"] == mode

        if mode == "W0":
            # In W0, index equals belief
            assert np.allclose(indices, details["v_belief"])
        elif mode == "W1":
            # In W1, index equals belief + recency
            assert np.allclose(indices, np.array(details["v_belief"]) + np.array(details["v_recency"]))


def test_deterministic_tie_breaking() -> None:
    """Verifies that equal indices break ties deterministically to the lowest frequency bin."""
    bands = [1e6 * i for i in range(10)]
    sched = WhittleScheduler(bands_hz=bands, ablation_mode="W0")
    obs = _make_dummy_obs(bin_idx=0, last_detection=False, timestamp=1.0, num_bins=10)

    # All initial beliefs in W0 are identical
    selected_bin = sched.select_bin(obs)
    assert selected_bin == 0, f"Expected deterministic tie-break to bin 0, got {selected_bin}"


def test_episode_reset_zero_leakage() -> None:
    """Verifies that reset clears all state, timers, and transition tables."""
    bands = [1e6 * i for i in range(10)]
    sched = WhittleScheduler(bands_hz=bands)

    # Push several transitions
    for step in range(1, 15):
        obs = _make_dummy_obs(bin_idx=step % 10, last_detection=(step % 2 == 0), timestamp=float(step), num_bins=10)
        sched.select_bin(obs)

    assert sched.belief_tracker.t_step > 0
    assert np.any(sched.transition_estimator.counts > 0)

    # Reset
    sched.reset()
    assert sched.belief_tracker.t_step == 0
    assert sched.belief_tracker.last_detected_bin is None
    assert np.all(sched.transition_estimator.counts == 0.0)
    assert np.all(sched.belief_tracker.get_uncertainties() == 1.0)
    assert sched.last_selected is None


def test_ground_truth_firewall() -> None:
    """Verifies that Whittle scheduler contains zero ground-truth keys or future attributes."""
    forbidden = {"emitter", "ground_truth", "gt", "snr", "carrier_frequency_hz", "transmitting"}
    bands = [1e6 * i for i in range(10)]
    sched = WhittleScheduler(bands_hz=bands)

    # Check internal state attributes
    for attr in dir(sched):
        assert attr.lower() not in forbidden, f"Forbidden attribute '{attr}' in WhittleScheduler"

    # Check explanation output
    obs = _make_dummy_obs(bin_idx=3, last_detection=True, timestamp=1.0, num_bins=10)
    sched.select_bin(obs)
    expl = sched.get_explanation()
    for k in expl:
        assert k.lower() not in forbidden, f"Forbidden key '{k}' in explanation"


def test_numerical_stability_and_edge_cases() -> None:
    """Verifies resilience against repeated misses, NaN guards, and empty observations."""
    bands = [1e6 * i for i in range(5)]
    sched = WhittleScheduler(bands_hz=bands)

    # 1. Repeated misses for 100 steps
    for step in range(1, 100):
        obs = _make_dummy_obs(bin_idx=step % 5, last_detection=False, timestamp=float(step), num_bins=5)
        chosen = sched.select_bin(obs)
        assert 0 <= chosen < 5

    # 2. Repeated detections in single bin
    for step in range(100, 150):
        obs = _make_dummy_obs(bin_idx=2, last_detection=True, timestamp=float(step), num_bins=5)
        chosen = sched.select_bin(obs)
        assert 0 <= chosen < 5

    # Check beliefs remain bounded
    beliefs = sched.belief_tracker.get_beliefs()
    assert np.all(beliefs >= 0.0)
    assert np.all(beliefs <= 1.0)
    assert not np.any(np.isnan(beliefs))


def test_factory_creation() -> None:
    """Verifies that factory creates WhittleScheduler cleanly by name."""
    bands = [1e6 * i for i in range(10)]
    sched1 = create_scheduler("whittle", bands)
    sched2 = create_scheduler("whittle_style", bands)

    assert isinstance(sched1, WhittleScheduler)
    assert isinstance(sched2, WhittleScheduler)
    assert sched1.num_bins == 10
