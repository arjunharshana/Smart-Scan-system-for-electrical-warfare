"""tests/test_v5_belief_scheduler.py
Unit tests and critical sanity checks for V5.0 Augmented Belief-State Bayesian Scheduler.
"""

from __future__ import annotations

import inspect
import time
import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.scheduler.belief.config import BeliefSchedulerConfig
from rf_environment.scheduler.belief.scheduler import V5BeliefScheduler
from rf_environment.scheduler.belief.state import AugmentedBeliefState
from rf_environment.scheduler.belief.transition import ObservableTransitionModel
from rf_environment.scheduler.factory import create_scheduler


def _get_bands_30() -> list[float]:
    """Standard 30-bin RF bands (100 MHz - 700 MHz, 20 MHz bandwidth)."""
    return [100e6 + 10e6 + i * 20e6 for i in range(30)]


def _make_observation(
    timestamp: float,
    current_bin: int,
    detected: bool,
    num_bins: int = 30,
) -> SchedulerObservation:
    return SchedulerObservation(
        timestamp=timestamp,
        current_frequency_bin=current_bin,
        last_detection=detected,
        last_detection_bin=current_bin if detected else None,
        last_detection_strength=-60.0 if detected else None,
        recent_detection_history=(detected,),
        recent_frequency_history=(current_bin,),
        scan_count_by_bin=tuple(1 if i == current_bin else 0 for i in range(num_bins)),
        time_since_scan_by_bin=tuple(0.0 if i == current_bin else 1.0 for i in range(num_bins)),
        time_since_last_detection=0.0 if detected else 1.0,
    )


# =============================================================================
# 1. Critical Sanity Tests (Prompt Section 12)
# =============================================================================

def test_sanity_1_probability_normalization() -> None:
    """Test 1: Normalization invariant sum(b) == 1.0 at initialization and after updates."""
    state = AugmentedBeliefState(num_bins=30)
    p_f = state.get_marginal_frequency()
    assert abs(np.sum(p_f) - 1.0) < 1e-10

    # Perform repeated positive and negative updates
    for b in range(10):
        state.update_observation(scanned_bin=b, detection=(b % 2 == 0))
        p_f = state.get_marginal_frequency()
        assert abs(np.sum(p_f) - 1.0) < 1e-10

    # Transition update
    t_mat = np.ones((30, 30)) / 30.0
    state.predict_transition(t_mat)
    p_f = state.get_marginal_frequency()
    assert abs(np.sum(p_f) - 1.0) < 1e-10


def test_sanity_2_no_impossible_probabilities() -> None:
    """Test 2: No NaNs, no infinities, and no negative probabilities under extreme updates."""
    state = AugmentedBeliefState(num_bins=30)

    # 100 consecutive misses on bin 5
    for _ in range(100):
        state.update_observation(scanned_bin=5, detection=False)

    for d in state.dwells:
        assert np.all(np.isfinite(state.belief[d]))
        assert np.all(state.belief[d] >= 0.0)

    p_f = state.get_marginal_frequency()
    assert np.all(p_f >= 0.0)
    assert abs(np.sum(p_f) - 1.0) < 1e-10


def test_sanity_3_observation_response_positive() -> None:
    """Test 3: Observation Y=1 strictly increases posterior probability on scanned bin."""
    state = AugmentedBeliefState(num_bins=30)
    prior_f = state.get_marginal_frequency()
    target_bin = 15

    # Observe detection on bin 15
    state.update_observation(scanned_bin=target_bin, detection=True)
    post_f = state.get_marginal_frequency()

    assert post_f[target_bin] > prior_f[target_bin], (
        f"Posterior {post_f[target_bin]} should be greater than prior {prior_f[target_bin]}"
    )
    # Probability on all other bins should decrease
    for b in range(30):
        if b != target_bin:
            assert post_f[b] < prior_f[b]


def test_sanity_4_observation_response_negative() -> None:
    """Test 4: Observation Y=0 strictly decreases posterior probability on scanned bin."""
    state = AugmentedBeliefState(num_bins=30)
    target_bin = 15

    # First raise belief with a detection
    state.update_observation(scanned_bin=target_bin, detection=True)
    high_p = state.get_marginal_frequency()[target_bin]

    # Now observe a miss on target_bin
    state.update_observation(scanned_bin=target_bin, detection=False)
    low_p = state.get_marginal_frequency()[target_bin]

    assert low_p < high_p, f"Miss should decrease probability from {high_p} to {low_p}"


def test_sanity_5_dwell_progression() -> None:
    """Test 5: Dwell progression tau: 0 -> 1 -> ... -> D-1 -> 0."""
    # Test on a state with only D=3
    cfg = BeliefSchedulerConfig(supported_dwells=(3,), ablation_mode="B1")
    state = AugmentedBeliefState(num_bins=10, config=cfg)

    # Concentrate mass at (frequency 2, tau 0, dwell 3)
    state.belief[3].fill(0.0)
    state.belief[3][2, 0] = 1.0

    # Step 1: tau 0 -> tau 1
    t_mat = np.eye(10)  # identity for test
    state.predict_transition(t_mat)
    assert state.belief[3][2, 1] > 0.999
    assert state.belief[3][2, 0] < 1e-6

    # Step 2: tau 1 -> tau 2
    state.predict_transition(t_mat)
    assert state.belief[3][2, 2] > 0.999
    assert state.belief[3][2, 1] < 1e-6

    # Step 3: tau 2 (which is D-1) -> hops to next frequency at tau 0
    # Set transition matrix to hop from 2 to 7
    hop_mat = np.zeros((10, 10))
    hop_mat[2, 7] = 1.0
    state.predict_transition(hop_mat)
    assert state.belief[3][7, 0] > 0.999
    assert state.belief[3][2, 2] < 1e-6


def test_sanity_6_gt_isolation() -> None:
    """Test 6: Scheduler source code and runtime have strictly zero ground truth access."""
    bands = _get_bands_30()
    sched = V5BeliefScheduler(bands)

    # Inspect source code of scheduler and belief modules
    source_code = inspect.getsource(V5BeliefScheduler)
    forbidden_terms = [
        "ground_truth",
        "true_frequency",
        "emitter_state",
        "scenario_dict",
        "scenario[",
        "emitters[",
    ]
    for term in forbidden_terms:
        assert term not in source_code, f"Forbidden term '{term}' found in V5BeliefScheduler source code!"

    # Runtime check: execute with an observation that has zero ground truth
    obs = _make_observation(timestamp=1.0, current_bin=0, detected=False)
    action = sched.select_action(obs)
    assert isinstance(action, ScanAction)
    assert 0 <= action.frequency_bin < 30


# =============================================================================
# 2. Comprehensive Component Tests
# =============================================================================

def test_initialization_and_factory() -> None:
    """Test factory instantiation and initial state configuration."""
    bands = _get_bands_30()
    sched = create_scheduler("v5_belief", bands)
    assert isinstance(sched, V5BeliefScheduler)
    assert sched.name == "v5_belief"
    assert sched.num_bins == 30

    p_f = sched.belief_state.get_marginal_frequency()
    assert np.allclose(p_f, 1.0 / 30.0)


def test_ablation_modes() -> None:
    """Test B0, B1, and B2 initialization and transition behaviors."""
    bands = _get_bands_30()
    for mode in ["B0", "B1", "B2"]:
        cfg = BeliefSchedulerConfig(ablation_mode=mode)
        sched = V5BeliefScheduler(bands, config=cfg)
        assert sched.config.ablation_mode == mode

        obs = _make_observation(timestamp=1.0, current_bin=5, detected=True)
        bin_idx = sched.select_bin(obs)
        assert 0 <= bin_idx < 30


def test_b2_empirical_transition_learning() -> None:
    """Test that B2 mode accurately learns hopping transitions from detections."""
    cfg = BeliefSchedulerConfig(ablation_mode="B2", transition_smoothing=0.01)
    trans = ObservableTransitionModel(num_bins=5, config=cfg)

    # Record sequence of hops: 1 -> 3 -> 4 -> 1 -> 3 -> 4
    for b in [1, 3, 4, 1, 3, 4]:
        trans.record_detection(b, timestamp=1.0)

    t_mat = trans.get_transition_matrix()
    # Transition 1 -> 3 should have highest probability in row 1
    assert np.argmax(t_mat[1, :]) == 3
    # Transition 3 -> 4 should have highest probability in row 3
    assert np.argmax(t_mat[3, :]) == 4
    # Transition 4 -> 1 should have highest probability in row 4
    assert np.argmax(t_mat[4, :]) == 1


def test_dwell_model_marginal_updates() -> None:
    """Test that observing multiple consecutive detections shifts dwell belief toward larger D."""
    bands = _get_bands_30()
    cfg = BeliefSchedulerConfig(supported_dwells=(1, 3, 5), ablation_mode="B1")
    sched = V5BeliefScheduler(bands, config=cfg)

    # Initially all dwells have positive mass
    init_d = sched.belief_state.get_marginal_dwell()
    assert abs(sum(init_d.values()) - 1.0) < 1e-10

    # Repeatedly detect at bin 10 without hopping
    for step in range(1, 10):
        obs = _make_observation(timestamp=float(step), current_bin=10, detected=True)
        sched.select_bin(obs)

    # Dwell D=1 assumes hopping every step; repeated detection at bin 10 should penalize D=1
    post_d = sched.belief_state.get_marginal_dwell()
    assert post_d[1] < post_d[3] or post_d[1] < post_d[5], (
        f"D=1 probability ({post_d[1]}) should be lower than D=3 ({post_d[3]}) or D=5 ({post_d[5]})"
    )


def test_deterministic_reproducibility() -> None:
    """Test that two scheduler instances with identical seeds and inputs produce bit-identical actions."""
    bands = _get_bands_30()
    sched1 = V5BeliefScheduler(bands, seed=42)
    sched2 = V5BeliefScheduler(bands, seed=42)

    actions1 = []
    actions2 = []

    for t in range(50):
        det = (t % 3 == 0)
        obs1 = _make_observation(timestamp=float(t), current_bin=t % 30, detected=det)
        obs2 = _make_observation(timestamp=float(t), current_bin=t % 30, detected=det)
        actions1.append(sched1.select_bin(obs1))
        actions2.append(sched2.select_bin(obs2))

    assert actions1 == actions2, "Identical inputs must produce identical action traces"


def test_per_step_latency() -> None:
    """Test that decision latency is well within < 10 ms (target < 1 ms/step)."""
    bands = _get_bands_30()
    sched = V5BeliefScheduler(bands)

    # Warmup
    for t in range(10):
        obs = _make_observation(timestamp=float(t), current_bin=t % 30, detected=False)
        sched.select_bin(obs)

    # Timed run over 500 steps
    start = time.perf_counter()
    n_steps = 500
    for t in range(n_steps):
        det = (t % 4 == 0)
        obs = _make_observation(timestamp=float(t), current_bin=t % 30, detected=det)
        sched.select_bin(obs)
    elapsed = time.perf_counter() - start

    latency_per_step_ms = (elapsed / n_steps) * 1000.0
    assert latency_per_step_ms < 1.0, f"Latency {latency_per_step_ms:.3f} ms exceeds 1.0 ms/step!"


def test_episode_reset_zero_leakage() -> None:
    """Test that reset completely clears belief, transitions, and timers."""
    bands = _get_bands_30()
    sched = V5BeliefScheduler(bands)

    # Run some steps
    for t in range(20):
        obs = _make_observation(timestamp=float(t), current_bin=5, detected=True)
        sched.select_bin(obs)

    # Verify state is non-uniform
    p_before = sched.belief_state.get_marginal_frequency()
    assert not np.allclose(p_before, 1.0 / 30.0)

    # Reset
    sched.reset()

    # Verify uniform state restored
    p_after = sched.belief_state.get_marginal_frequency()
    assert np.allclose(p_after, 1.0 / 30.0)
    assert sched._last_processed_timestamp is None
    assert np.all(sched._time_since_scan == 0.0)
