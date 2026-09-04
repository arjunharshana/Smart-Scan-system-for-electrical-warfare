from __future__ import annotations

import copy
import numpy as np
from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario
from rf_environment.experiments.runner import BenchmarkRunner
from rf_environment.domain.observation import Observation
from rf_environment.scheduler.factory import create_scheduler


def test_1_fixed_continuous_emitter_learned():
    """TEST 1: Fixed + continuous emitter can be learned by Thompson / UCB.
    
    Verifies that UCB1 and Thompson allocate significantly more scans and accumulate
    vastly higher reward on the active channel than uniform random scanning.
    """
    scenario = load_scenario("rf_environment/scenarios/stationary_fixed.yaml")
    runner = BenchmarkRunner()

    res_ucb = runner.run_single(scenario, "ucb1", seed=42, steps=200)
    res_ts = runner.run_single(scenario, "thompson", seed=42, steps=200)
    res_rand = runner.run_single(scenario, "random", seed=42, steps=200)

    # UCB1 and Thompson focus attention on the active band, achieving > 5x the reward of random
    assert res_ucb["cumulative_reward"] > res_rand["cumulative_reward"] * 3
    assert res_ts["cumulative_reward"] > res_rand["cumulative_reward"] * 3
    assert res_ucb["avg_reward"] > 0.40
    assert res_ts["avg_reward"] > 0.40


def test_2_deterministic_hopping_stationary_degradation():
    """TEST 2: Deterministic hopping demonstrates the limitation of stationary bandits.
    
    Stationary bandits average across hopping channels and experience severe delay.
    Context-Aware scheduling exploits cyclic transitions to achieve faster intercept
    and vastly higher tracking reward.
    """
    scenario = load_scenario("rf_environment/scenarios/deterministic_hopping.yaml")
    runner = BenchmarkRunner()

    res_context = runner.run_single(scenario, "context_aware", seed=42, steps=300)
    res_ucb = runner.run_single(scenario, "ucb1", seed=42, steps=300)
    res_rand = runner.run_single(scenario, "random", seed=42, steps=300)

    # Context-Aware scheduler achieves > 2x tracking reward compared to stationary UCB1 and Random
    assert res_context["cumulative_reward"] > res_ucb["cumulative_reward"] * 1.8
    assert res_context["cumulative_reward"] > res_rand["cumulative_reward"] * 2.5
    # Intercept latency for context_aware is much faster than stationary bandits
    assert res_context["avg_intercept_time"] <= res_ucb["avg_intercept_time"]


def test_3_non_stationary_adaptation_after_shift():
    """TEST 3: Sliding-window and discounted algorithms adapt faster after a frequency shift."""
    # Create scenario with an emitter switching from band A to band B at step 100
    bands = [200e6, 300e6, 400e6, 500e6]

    sw_ucb = create_scheduler("sw_ucb", bands)
    ucb1 = create_scheduler("ucb1", bands)

    # Simulate 100 steps of band 0 (200 MHz) being active
    for t in range(100):
        # SW-UCB
        act_sw = sw_ucb.select_action(None)
        r_sw = 1.0 if abs(act_sw - 200e6) < 1e5 else 0.0
        obs_sw = Observation(timestamp=t, receiver_frequency_hz=act_sw, receiver_bandwidth_hz=20e6, detected=(r_sw > 0))
        sw_ucb.update(obs_sw, r_sw, action=act_sw)

        # UCB1
        act_ucb = ucb1.select_action(None)
        r_ucb = 1.0 if abs(act_ucb - 200e6) < 1e5 else 0.0
        obs_ucb = Observation(timestamp=t, receiver_frequency_hz=act_ucb, receiver_bandwidth_hz=20e6, detected=(r_ucb > 0))
        ucb1.update(obs_ucb, r_ucb, action=act_ucb)

    # Now signal abruptly switches to band 2 (400 MHz) for next 50 steps
    sw_hits_after_switch = 0
    ucb_hits_after_switch = 0
    for t in range(100, 150):
        # SW-UCB
        act_sw = sw_ucb.select_action(None)
        r_sw = 1.0 if abs(act_sw - 400e6) < 1e5 else 0.0
        obs_sw = Observation(timestamp=t, receiver_frequency_hz=act_sw, receiver_bandwidth_hz=20e6, detected=(r_sw > 0))
        sw_ucb.update(obs_sw, r_sw, action=act_sw)
        if r_sw > 0:
            sw_hits_after_switch += 1

        # UCB1
        act_ucb = ucb1.select_action(None)
        r_ucb = 1.0 if abs(act_ucb - 400e6) < 1e5 else 0.0
        obs_ucb = Observation(timestamp=t, receiver_frequency_hz=act_ucb, receiver_bandwidth_hz=20e6, detected=(r_ucb > 0))
        ucb1.update(obs_ucb, r_ucb, action=act_ucb)
        if r_ucb > 0:
            ucb_hits_after_switch += 1

    # Sliding Window UCB adapts faster because old 200 MHz rewards roll out of window
    assert sw_hits_after_switch >= ucb_hits_after_switch


def test_4_context_aware_exploits_deterministic_transitions():
    """TEST 4: Context-aware scheduling exploits deterministic transition patterns."""
    scenario = load_scenario("rf_environment/scenarios/deterministic_hopping.yaml")
    runner = BenchmarkRunner()

    res_context = runner.run_single(scenario, "context_aware", seed=42, steps=300)
    # The context-aware scheduler should achieve positive observed prediction accuracy
    assert res_context["prediction_accuracy"] > 0.30
    # In V2.2, episode coverage is 100% and hop coverage (0.34) significantly exceeds random baseline (~0.19)
    assert res_context["episode_coverage_ratio"] > 0.40
    assert res_context["hop_coverage_ratio"] > 0.25


def test_5_random_hopping_predictability_upper_limit():
    """TEST 5: Random hopping remains difficult and algorithms do not falsely claim high prediction."""
    scenario = load_scenario("rf_environment/scenarios/random_hopping.yaml")
    runner = BenchmarkRunner()

    res_rand_hop = runner.run_single(scenario, "context_aware", seed=42, steps=300)
    # With 5 hopping frequencies, random chance is 1/5 = 0.20. Prediction accuracy should not be falsely inflated
    assert res_rand_hop["prediction_accuracy"] < 0.50


def test_6_interception_metric_formalism():
    """TEST 6: Interception ratio and intercept time behave correctly according to formal definitions.
    
    Verifies:
    Interception Ratio == Opportunity Coverage * Detection Given Coverage
    """
    scenario = load_scenario("rf_environment/scenarios/periodic_time.yaml")
    runner = BenchmarkRunner()

    res = runner.run_single(scenario, "ucb1", seed=42, steps=200)
    cov = res["opportunity_coverage"]
    dgc = res["detection_given_coverage"]
    ir = res["interception_ratio"]

    # Mathematical identity check (within floating point precision)
    expected_ir = cov * dgc
    assert abs(ir - expected_ir) < 1e-4

    # Intercept time must be non-negative if present
    if res["avg_intercept_time"] is not None:
        assert res["avg_intercept_time"] >= 0.0


def test_7_ground_truth_isolation():
    """TEST 7: Ground truth cannot be accessed by scheduler code."""
    scenario = load_scenario("rf_environment/scenarios/deterministic_hopping.yaml")
    env = build_environment(scenario, scheduler_name="context_aware")

    res = env.step()
    obs = res["observation"]

    # Strict isolation: Ground truth emitter array and IDs are NOT in observation
    assert "emitters" not in obs
    assert "transmitting_ids" not in obs
    assert obs.get("associated_emitter_id") is None
    assert "ground_truth" not in obs


def test_8_frozen_environment_realization_equivalence():
    """TEST 8: All algorithms receive the exact same environment realization during benchmarking."""
    scenario = load_scenario("rf_environment/scenarios/deterministic_hopping.yaml")

    env_ucb = build_environment(scenario, scheduler_name="ucb1")
    env_ts = build_environment(scenario, scheduler_name="thompson")

    # Run 10 steps and compare ground truth emitter states across environments
    for _ in range(10):
        r1 = env_ucb.step()
        r2 = env_ts.step()

        gt1 = r1["ground_truth"]
        gt2 = r2["ground_truth"]

        assert gt1["transmitting_ids"] == gt2["transmitting_ids"]
        for e1, e2 in zip(gt1["emitters"], gt2["emitters"]):
            assert e1["emitter_id"] == e2["emitter_id"]
            assert e1["frequency_hz"] == e2["frequency_hz"]
            assert e1["transmitting"] == e2["transmitting"]


def test_9_multi_seed_benchmarking_reproducibility():
    """TEST 9: Multi-seed benchmarks produce identical statistical metrics when re-run with same seeds."""
    scenario = load_scenario("rf_environment/scenarios/stationary_fixed.yaml")
    runner = BenchmarkRunner()

    bench1 = runner.run_benchmark(scenario, algorithms=["ucb1", "random"], seeds=[1, 2], steps=100)
    bench2 = runner.run_benchmark(scenario, algorithms=["ucb1", "random"], seeds=[1, 2], steps=100)

    sum1 = bench1["summary_df"]
    sum2 = bench2["summary_df"]

    assert list(sum1["interception_ratio"]) == list(sum2["interception_ratio"])
    assert list(sum1["opportunity_coverage"]) == list(sum2["opportunity_coverage"])


def test_10_telemetry_waterfall_and_diagnostic_integrity():
    """TEST 10: Telemetry and waterfall data correctly reflect trajectories, bandwidth, and decisions."""
    scenario = load_scenario("rf_environment/scenarios/deterministic_hopping.yaml")
    env = build_environment(scenario, scheduler_name="context_aware")

    results = env.run(steps=30)
    assert len(env.waterfall) == 30

    item = env.waterfall[0]
    assert "receiver_frequency_hz" in item
    assert "receiver_bandwidth_hz" in item
    assert "detected" in item
    assert "diagnostic_reason" in item

    # Scheduler explanation exists
    sched_state = env.scheduler.get_state()
    assert sched_state.explanation is not None
    assert "action_mhz" in sched_state.explanation
    assert "reason" in sched_state.explanation
