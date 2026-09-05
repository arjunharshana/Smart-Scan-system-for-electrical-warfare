from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario
from rf_environment.experiments.runner import BenchmarkRunner
from rf_environment.scheduler.rl.ddqn_scheduler import DDQNScheduler
from rf_environment.scheduler.rl.encoder import ObservationEncoder
from rf_environment.scheduler.rl.temporal_encoder import TemporalObservationEncoder


TRAIN_SEEDS = [42, 123, 456, 789]
EVAL_SEEDS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

# Base Spectrum: 100 MHz - 700 MHz, 20 MHz rx bw -> 30 bins
SPECTRUM_CFG = {"min_frequency_hz": 100_000_000, "max_frequency_hz": 700_000_000}
RECEIVER_CFG = {
    "instantaneous_bandwidth_hz": 20_000_000,
    "sensitivity_dbm": -90,
    "noise_floor_dbm": -100,
    "detection_threshold_db": 6,
    "tuning_time_ms": 0,
}
DETECTOR_CFG = {"p_detection": 0.95, "p_false_alarm": 0.02}

def bin_to_center_hz(b: int) -> float:
    return 100_000_000.0 + 10_000_000.0 + float(b) * 20_000_000.0

def make_scenario(
    emitters: list[dict[str, Any]],
    total_time_steps: int = 300,
    seed: int = 42,
) -> dict[str, Any]:
    return {
        "simulation": {
            "total_time_steps": total_time_steps,
            "time_step_ms": 10,
            "seed": seed,
        },
        "spectrum": SPECTRUM_CFG,
        "receiver": RECEIVER_CFG,
        "detector": DETECTOR_CFG,
        "scheduler": {"type": "sequential"},
        "emitters": emitters,
    }


def evaluate_scheduler(
    env_scenario: dict[str, Any],
    scheduler: Any,
    seed: int,
    steps: int = 300,
) -> dict[str, float]:
    scenario_copy = copy.deepcopy(env_scenario)
    scenario_copy["simulation"]["seed"] = seed
    scenario_copy["simulation"]["total_time_steps"] = steps

    env = build_environment(scenario_copy)
    scheduler.eval()
    scheduler.epsilon = 0.0

    num_bins = len(env.bands_hz)
    class SafeSchedulerWrapper:
        def __init__(self, inner: Any, max_b: int) -> None:
            self._inner = inner
            self._max_b = max_b
        def select_action(self, obs: SchedulerObservation) -> ScanAction:
            act = self._inner.select_action(obs)
            b = act.frequency_bin if isinstance(act, ScanAction) else int(act)
            return ScanAction(frequency_bin=min(max(b, 0), self._max_b - 1))
        def update_policy(self, *args: Any, **kwargs: Any) -> None:
            pass
        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    env.scheduler = SafeSchedulerWrapper(scheduler, num_bins)

    res = env.run(steps=steps)
    snap = env.metrics.snapshot(env.clock.time_step).to_dict()

    hits = sum(1 for r in res if r.observation.last_detection)
    det_rate = float(hits) / float(len(res)) if res else 0.0

    return {
        "interception_ratio": float(snap.get("interception_ratio", 0.0)),
        "opportunity_coverage": float(snap.get("opportunity_coverage", 0.0)),
        "detection_given_coverage": float(snap.get("detection_given_coverage", 0.0)),
        "detection_rate": det_rate,
        "total_steps": len(res),
    }


def evaluate_baseline(
    runner: BenchmarkRunner,
    scenario_dict: dict[str, Any],
    scheduler_name: str,
    seeds: list[int],
    steps: int = 300,
) -> dict[str, Any]:
    irs, covs, det_covs, det_rates = [], [], [], []
    for s in seeds:
        res = runner.run_single(
            scenario=scenario_dict,
            scheduler_name=scheduler_name,
            seed=s,
            steps=steps,
        )
        irs.append(res["interception_ratio"])
        covs.append(res["opportunity_coverage"])
        det_covs.append(res["detection_given_coverage"])
        det_rates.append(res.get("detection_rate", res["interception_ratio"]))

    return {
        "ir_mean": float(np.mean(irs)),
        "ir_std": float(np.std(irs)),
        "cov_mean": float(np.mean(covs)),
        "det_cov_mean": float(np.mean(det_covs)),
        "det_rate_mean": float(np.mean(det_rates)),
        "per_seed_ir": irs,
    }


def main() -> None:
    print("=" * 80)
    print("🚀 SIH26055 — V3.1 DDQN ROBUSTNESS & TEMPORAL REPRESENTATION STUDY")
    print("=" * 80)

    runner = BenchmarkRunner()
    results_payload: dict[str, Any] = {}

    # Load canonical scenarios
    scenarios = {
        "deterministic_hopping": load_scenario("rf_environment/scenarios/deterministic_hopping.yaml"),
        "random_hopping": load_scenario("rf_environment/scenarios/random_hopping.yaml"),
        "periodic_time": load_scenario("rf_environment/scenarios/periodic_time.yaml"),
    }

    # =========================================================================
    # STEP 1: REPRODUCE V3.0 BASELINE
    # =========================================================================
    print("\n--- STEP 1: Reproducing V3.0 Baseline ---")
    v3_baseline: dict[str, Any] = {}

    # Train standard V3.0 DDQN (Encoder A) per scenario
    for sc_name, sc_cfg in scenarios.items():
        print(f"   [V3.0 Baseline] Training scenario: {sc_name}...")
        sub_models = []
        train_rewards = []
        train_losses = []

        for t_seed in TRAIN_SEEDS:
            env = build_environment(copy.deepcopy(sc_cfg))
            num_bins = len(env.bands_hz)
            enc = ObservationEncoder(num_bins=num_bins)
            sched = DDQNScheduler(
                bands_hz=env.bands_hz,
                encoder=enc,
                seed=t_seed,
            )
            sched.train()
            env.scheduler = sched

            for ep in range(4):
                ep_seed = t_seed + ep * 1000
                saved_eps = sched.epsilon
                env.reset(seed=ep_seed)
                sched.epsilon = saved_eps
                res = env.run(steps=300)
                train_rewards.append(float(np.mean([r.reward for r in res])))

            train_losses.append(float(np.mean(sched.losses[-50:])) if sched.losses else 0.0)
            sub_models.append(sched)

        # Evaluate V3.0 DDQN across evaluation seeds
        eval_irs = []
        for e_seed in EVAL_SEEDS:
            seed_irs = []
            for model in sub_models:
                ev = evaluate_scheduler(sc_cfg, model, seed=e_seed, steps=300)
                seed_irs.append(ev["interception_ratio"])
            eval_irs.append(float(np.mean(seed_irs)))

        # Also get Context-Aware baseline
        ca_eval = evaluate_baseline(runner, sc_cfg, "context_aware", EVAL_SEEDS, steps=300)

        v3_baseline[sc_name] = {
            "ddqn_ir_mean": float(np.mean(eval_irs)),
            "ddqn_ir_std": float(np.std(eval_irs)),
            "ca_ir_mean": ca_eval["ir_mean"],
            "ca_ir_std": ca_eval["ir_std"],
            "train_loss_mean": float(np.mean(train_losses)),
            "train_reward_mean": float(np.mean(train_rewards)),
            "per_seed_ir": eval_irs,
        }
        print(f"      -> DDQN IR = {v3_baseline[sc_name]['ddqn_ir_mean']:.4f} ± {v3_baseline[sc_name]['ddqn_ir_std']:.4f} | CA IR = {ca_eval['ir_mean']:.4f}")

    # Baseline OOD Test
    print("   [V3.0 Baseline] Evaluating baseline OOD hopping test...")
    ood_cfg = copy.deepcopy(scenarios["deterministic_hopping"])
    for em in ood_cfg.get("emitters", []):
        if em.get("id") == "E_DETERM_HOP":
            em["frequency_behavior"]["frequencies_hz"] = [300_000_000, 500_000_000, 650_000_000]
            em["frequency_behavior"]["dwell_steps"] = 5

    # Models from deterministic_hopping
    det_models = []
    for t_seed in TRAIN_SEEDS:
        env = build_environment(copy.deepcopy(scenarios["deterministic_hopping"]))
        sched = DDQNScheduler(bands_hz=env.bands_hz, seed=t_seed)
        sched.train()
        env.scheduler = sched
        for ep in range(4):
            saved_eps = sched.epsilon
            env.reset(seed=t_seed + ep * 1000)
            sched.epsilon = saved_eps
            env.run(steps=300)
        det_models.append(sched)

    ddqn_ood_irs = []
    for e_seed in EVAL_SEEDS:
        sub_irs = [evaluate_scheduler(ood_cfg, m, seed=e_seed, steps=300)["interception_ratio"] for m in det_models]
        ddqn_ood_irs.append(float(np.mean(sub_irs)))
    ca_ood = evaluate_baseline(runner, ood_cfg, "context_aware", EVAL_SEEDS, steps=300)

    v3_baseline["ood_hopping"] = {
        "ddqn_ir_mean": float(np.mean(ddqn_ood_irs)),
        "ddqn_ir_std": float(np.std(ddqn_ood_irs)),
        "ca_ir_mean": ca_ood["ir_mean"],
        "ca_ir_std": ca_ood["ir_std"],
    }
    print(f"      -> Baseline OOD: DDQN IR = {v3_baseline['ood_hopping']['ddqn_ir_mean']:.4f} | CA IR = {ca_ood['ir_mean']:.4f}")
    results_payload["v3_0_baseline"] = v3_baseline

    # =========================================================================
    # STEP 2: ENCODER COMPARISON (EXPERIMENTS 1, 2, 3)
    # =========================================================================
    print("\n--- STEP 2: Encoder Comparison (Encoders A, B, C, D) ---")
    encoder_configs = {
        "Encoder_A_Baseline": lambda nb: TemporalObservationEncoder.create_encoder_a(nb),
        "Encoder_B_Temporal": lambda nb: TemporalObservationEncoder.create_encoder_b(nb),
        "Encoder_C_NoAbsTime": lambda nb: TemporalObservationEncoder.create_encoder_c(nb),
        "Encoder_D_Fourier": lambda nb: TemporalObservationEncoder.create_encoder_d(nb),
    }

    encoder_comparison_results: dict[str, Any] = {}
    det_scenario = scenarios["deterministic_hopping"]

    for enc_name, enc_builder in encoder_configs.items():
        print(f"   [Encoder Study] Training with {enc_name} on deterministic_hopping...")
        enc_models = []
        for t_seed in TRAIN_SEEDS:
            env = build_environment(copy.deepcopy(det_scenario))
            num_bins = len(env.bands_hz)
            enc = enc_builder(num_bins)
            sched = DDQNScheduler(
                bands_hz=env.bands_hz,
                encoder=enc,
                seed=t_seed,
            )
            sched.train()
            env.scheduler = sched
            for ep in range(4):
                saved_eps = sched.epsilon
                env.reset(seed=t_seed + ep * 1000)
                sched.epsilon = saved_eps
                env.run(steps=300)
            enc_models.append(sched)

        # In-Distribution evaluation
        id_irs = []
        for e_seed in EVAL_SEEDS:
            s_irs = [evaluate_scheduler(det_scenario, m, seed=e_seed, steps=300)["interception_ratio"] for m in enc_models]
            id_irs.append(float(np.mean(s_irs)))

        # OOD evaluation
        ood_irs = []
        for e_seed in EVAL_SEEDS:
            s_irs = [evaluate_scheduler(ood_cfg, m, seed=e_seed, steps=300)["interception_ratio"] for m in enc_models]
            ood_irs.append(float(np.mean(s_irs)))

        encoder_comparison_results[enc_name] = {
            "id_ir_mean": float(np.mean(id_irs)),
            "id_ir_std": float(np.std(id_irs)),
            "ood_ir_mean": float(np.mean(ood_irs)),
            "ood_ir_std": float(np.std(ood_irs)),
            "retention": float(np.mean(ood_irs)) / max(float(np.mean(id_irs)), 1e-6),
        }
        print(f"      -> {enc_name}: ID IR = {np.mean(id_irs):.4f} | OOD IR = {np.mean(ood_irs):.4f} (Retention = {encoder_comparison_results[enc_name]['retention']:.3f})")

    results_payload["encoder_comparison"] = encoder_comparison_results

    # =========================================================================
    # STEP 3: DIVERSIFIED TRAINING DISTRIBUTION GENERATOR (EXPERIMENT 4 & 8)
    # =========================================================================
    print("\n--- STEP 3: Diversified Training Distribution & Replay Profiling ---")

    # Hopping sequences:
    # 30 bins: 0..29
    # Training patterns (Seen structures):
    # Train Pool has structured hopping subsets including [5, 15, 25] (bins for 200, 400, 600 MHz):
    # Held out completely for OOD testing: [10, 20, 27] (300, 500, 650 MHz)
    train_hopping_subsets = [
        [5, 15, 25],
        [3, 11, 19],
        [2, 12, 22],
        [7, 17, 23],
        [4, 14, 24],
    ]

    # Replay buffer profiling containers
    replay_transitions = set()
    replay_sequences = set()
    replay_scenario_types = set()
    replay_active_channel_sets = set()

    def generate_training_scenario(episode_idx: int, rng: np.random.Generator) -> tuple[dict[str, Any], str]:
        # Distribution:
        # 0..24: 25% stationary
        # 25..59: 35% deterministic hopping
        # 60..79: 20% random hopping
        # 80..89: 10% periodic/burst
        # 90..99: 10% mixed/regime-shift
        p_val = rng.uniform(0, 100)

        if p_val < 25.0:
            # Stationary
            b = rng.integers(0, 30)
            em = {
                "id": "E_STAT",
                "type": "radar",
                "frequency_behavior": {"type": "fixed", "frequency_hz": bin_to_center_hz(b)},
                "time_behavior": {"type": "continuous"},
                "power_dbm": float(rng.uniform(-25.0, -15.0)),
                "bandwidth_hz": 2_000_000,
            }
            sc_type = "stationary"
            replay_scenario_types.add(sc_type)
            replay_active_channel_sets.add(tuple([b]))
            return make_scenario([em], total_time_steps=300), sc_type

        elif p_val < 60.0:
            # Deterministic Hopping
            subset = copy.copy(train_hopping_subsets[rng.integers(0, len(train_hopping_subsets))])
            if rng.random() < 0.5:
                rng.shuffle(subset)
            dwell = int(rng.choice([1, 2, 3]))
            freqs = [bin_to_center_hz(b) for b in subset]
            em = {
                "id": "E_HOP",
                "type": "radar",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": freqs,
                    "mode": "sequential",
                    "dwell_steps": dwell,
                },
                "time_behavior": {"type": "continuous"},
                "power_dbm": float(rng.uniform(-22.0, -16.0)),
                "bandwidth_hz": 2_000_000,
            }
            sc_type = "deterministic_hopping"
            replay_scenario_types.add(sc_type)
            replay_sequences.add(tuple(subset))
            replay_active_channel_sets.add(tuple(sorted(subset)))
            for i in range(len(subset)):
                replay_transitions.add((subset[i], subset[(i + 1) % len(subset)]))
            return make_scenario([em], total_time_steps=300), sc_type

        elif p_val < 80.0:
            # Random Hopping
            subset = copy.copy(train_hopping_subsets[rng.integers(0, len(train_hopping_subsets))])
            dwell = int(rng.choice([1, 2]))
            freqs = [bin_to_center_hz(b) for b in subset]
            em = {
                "id": "E_RAND_HOP",
                "type": "communication",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": freqs,
                    "mode": "random",
                    "dwell_steps": dwell,
                },
                "time_behavior": {"type": "continuous"},
                "power_dbm": float(rng.uniform(-22.0, -16.0)),
                "bandwidth_hz": 2_000_000,
            }
            sc_type = "random_hopping"
            replay_scenario_types.add(sc_type)
            replay_active_channel_sets.add(tuple(sorted(subset)))
            return make_scenario([em], total_time_steps=300), sc_type

        elif p_val < 90.0:
            # Periodic Burst
            b = rng.integers(0, 30)
            period = int(rng.choice([8, 10, 12, 16]))
            active = int(rng.choice([2, 3, 4]))
            em = {
                "id": "E_PERIODIC",
                "type": "radar",
                "frequency_behavior": {"type": "fixed", "frequency_hz": bin_to_center_hz(b)},
                "time_behavior": {
                    "type": "periodic",
                    "on_duration": active,
                    "off_duration": max(period - active, 1),
                    "phase": int(rng.integers(0, period)),
                },
                "power_dbm": float(rng.uniform(-20.0, -15.0)),
                "bandwidth_hz": 2_000_000,
            }
            sc_type = "periodic_burst"
            replay_scenario_types.add(sc_type)
            replay_active_channel_sets.add(tuple([b]))
            return make_scenario([em], total_time_steps=300), sc_type

        else:
            # Mixed / Regime Shift
            b1 = rng.integers(0, 15)
            b2 = rng.integers(15, 30)
            em1 = {
                "id": "E_REGIME_1",
                "type": "radar",
                "frequency_behavior": {"type": "fixed", "frequency_hz": bin_to_center_hz(b1)},
                "time_behavior": {"type": "continuous"},
                "power_dbm": -18.0,
                "bandwidth_hz": 2_000_000,
            }
            em2 = {
                "id": "E_REGIME_2",
                "type": "radar",
                "frequency_behavior": {"type": "fixed", "frequency_hz": bin_to_center_hz(b2)},
                "time_behavior": {"type": "continuous"},
                "power_dbm": -18.0,
                "bandwidth_hz": 2_000_000,
            }
            sc_type = "mixed_regime"
            replay_scenario_types.add(sc_type)
            replay_active_channel_sets.add(tuple([b1, b2]))
            return make_scenario([em1, em2], total_time_steps=300), sc_type

    # Train diversified models with Encoder B (Temporal)
    print("   Training Diversified DDQN across seeds...")
    diversified_models = []
    total_episodes_planned = 12  # 12 episodes * 300 steps = 3600 steps

    for t_seed in TRAIN_SEEDS:
        rng = np.random.default_rng(t_seed)
        dummy_sc, _ = generate_training_scenario(0, rng)
        env = build_environment(dummy_sc)
        num_bins = len(env.bands_hz)
        enc = TemporalObservationEncoder.create_encoder_b(num_bins=num_bins)
        sched = DDQNScheduler(
            bands_hz=env.bands_hz,
            encoder=enc,
            seed=t_seed,
        )
        sched.train()

        for ep in range(total_episodes_planned):
            ep_sc, _ = generate_training_scenario(ep, rng)
            env_ep = build_environment(ep_sc)
            sched.bands_hz = env_ep.bands_hz
            saved_eps = sched.epsilon
            env_ep.scheduler = sched
            env_ep.reset(seed=t_seed + ep * 500)
            sched.epsilon = saved_eps
            env_ep.run(steps=300)

        diversified_models.append(sched)

    replay_profile = {
        "unique_frequency_transitions": len(replay_transitions),
        "unique_hopping_sequences": len(replay_sequences),
        "number_of_scenario_types": len(replay_scenario_types),
        "distinct_active_channel_sets": len(replay_active_channel_sets),
    }
    print("   Replay Buffer Diversity Profile:", replay_profile)
    results_payload["replay_diversity_profile"] = replay_profile

    # =========================================================================
    # STEP 4: GENERALIZATION MATRIX (EXPERIMENT 5 & 6)
    # =========================================================================
    print("\n--- STEP 4: Generalization Matrix Evaluation ---")

    # Define the 6 conditions:
    # A. Seen Structure: [5, 15, 25] (200, 400, 600 MHz) with dwell=3 (exact deterministic_hopping scenario)
    sc_A = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]], "mode": "sequential", "dwell_steps": 3},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # B. Unseen Permutation: [25, 5, 15] with dwell=3 (same channels, reversed/permuted order)
    sc_B = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [25, 5, 15]], "mode": "sequential", "dwell_steps": 3},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # C. Unseen Phase: [15, 25, 5] with dwell=3 (same sequence, starting at offset phase)
    sc_C = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [15, 25, 5]], "mode": "sequential", "dwell_steps": 3},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # D. Unseen Dwell: [5, 15, 25] with dwell=1 (fast hop)
    sc_D = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]], "mode": "sequential", "dwell_steps": 1},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # E. Unseen Active Subset: Completely held out channels: [10, 20, 27] (300, 500, 650 MHz) with dwell=3
    sc_E = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [10, 20, 27]], "mode": "sequential", "dwell_steps": 3},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # F. Mixed Shift: Unseen channels [10, 27, 20], dwell=5 (the exact V3.0 OOD benchmark test)
    sc_F = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [10, 27, 20]], "mode": "sequential", "dwell_steps": 5},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    conditions = {
        "A_Seen_Structure": sc_A,
        "B_Unseen_Permutation": sc_B,
        "C_Unseen_Phase": sc_C,
        "D_Unseen_Dwell": sc_D,
        "E_Unseen_Subset": sc_E,
        "F_Mixed_Shift": sc_F,
    }

    # Evaluate: V3.0 DDQN, Best V3.1 DDQN (Diversified), Context-Aware
    generalization_matrix: dict[str, Any] = {}

    for cond_name, cond_sc in conditions.items():
        print(f"   Evaluating Condition: {cond_name}...")
        # Context-Aware
        res_ca = evaluate_baseline(runner, cond_sc, "context_aware", EVAL_SEEDS, steps=300)

        # V3.0 DDQN (det_models trained on fixed pattern)
        v3_irs = []
        for e_s in EVAL_SEEDS:
            s_irs = [evaluate_scheduler(cond_sc, m, seed=e_s, steps=300)["interception_ratio"] for m in det_models]
            v3_irs.append(float(np.mean(s_irs)))

        # V3.1 Diversified DDQN
        v31_irs = []
        v31_covs = []
        v31_det_covs = []
        v31_det_rates = []
        for e_s in EVAL_SEEDS:
            s_irs, s_covs, s_det_covs, s_det_rates = [], [], [], []
            for m in diversified_models:
                ev = evaluate_scheduler(cond_sc, m, seed=e_s, steps=300)
                s_irs.append(ev["interception_ratio"])
                s_covs.append(ev["opportunity_coverage"])
                s_det_covs.append(ev["detection_given_coverage"])
                s_det_rates.append(ev["detection_rate"])
            v31_irs.append(float(np.mean(s_irs)))
            v31_covs.append(float(np.mean(s_covs)))
            v31_det_covs.append(float(np.mean(s_det_covs)))
            v31_det_rates.append(float(np.mean(s_det_rates)))

        generalization_matrix[cond_name] = {
            "context_aware": {
                "ir_mean": res_ca["ir_mean"],
                "ir_std": res_ca["ir_std"],
                "cov_mean": res_ca["cov_mean"],
            },
            "v3_0_ddqn": {
                "ir_mean": float(np.mean(v3_irs)),
                "ir_std": float(np.std(v3_irs)),
            },
            "v3_1_ddqn": {
                "ir_mean": float(np.mean(v31_irs)),
                "ir_std": float(np.std(v31_irs)),
                "cov_mean": float(np.mean(v31_covs)),
                "det_given_cov": float(np.mean(v31_det_covs)),
                "detection_rate": float(np.mean(v31_det_rates)),
                "retention_vs_seen": float(np.mean(v31_irs)) / max(generalization_matrix.get("A_Seen_Structure", {}).get("v3_1_ddqn", {}).get("ir_mean", float(np.mean(v31_irs))), 1e-6),
            },
        }
        print(f"      {cond_name} -> CA: {res_ca['ir_mean']:.4f} | V3.0 DDQN: {np.mean(v3_irs):.4f} | V3.1 DDQN: {np.mean(v31_irs):.4f}")

    results_payload["generalization_matrix"] = generalization_matrix

    # =========================================================================
    # STEP 5: TRAINING DATA QUANTITY ABLATION (EXPERIMENT 7)
    # =========================================================================
    print("\n--- STEP 5: Training Data Quantity Ablation (25%, 50%, 100%) ---")
    data_quantities = {
        "25_percent": 3,   # 3 episodes
        "50_percent": 6,   # 6 episodes
        "100_percent": 12, # 12 episodes
    }

    quantity_results: dict[str, Any] = {}
    for q_name, ep_count in data_quantities.items():
        q_models = []
        for t_seed in TRAIN_SEEDS:
            rng = np.random.default_rng(t_seed)
            dummy_sc, _ = generate_training_scenario(0, rng)
            env = build_environment(dummy_sc)
            enc = TemporalObservationEncoder.create_encoder_b(len(env.bands_hz))
            sched = DDQNScheduler(bands_hz=env.bands_hz, encoder=enc, seed=t_seed)
            sched.train()
            for ep in range(ep_count):
                ep_sc, _ = generate_training_scenario(ep, rng)
                env_ep = build_environment(ep_sc)
                saved_eps = sched.epsilon
                env_ep.scheduler = sched
                env_ep.reset(seed=t_seed + ep * 500)
                sched.epsilon = saved_eps
                env_ep.run(steps=300)
            q_models.append(sched)

        # In-dist (Condition A) vs OOD (Condition E)
        id_irs = [float(np.mean([evaluate_scheduler(sc_A, m, seed=s, steps=300)["interception_ratio"] for m in q_models])) for s in EVAL_SEEDS]
        ood_irs = [float(np.mean([evaluate_scheduler(sc_E, m, seed=s, steps=300)["interception_ratio"] for m in q_models])) for s in EVAL_SEEDS]

        quantity_results[q_name] = {
            "episodes": ep_count,
            "id_ir_mean": float(np.mean(id_irs)),
            "id_ir_std": float(np.std(id_irs)),
            "ood_ir_mean": float(np.mean(ood_irs)),
            "ood_ir_std": float(np.std(ood_irs)),
            "retention": float(np.mean(ood_irs)) / max(float(np.mean(id_irs)), 1e-6),
        }
        print(f"   {q_name} ({ep_count} eps): ID IR = {np.mean(id_irs):.4f} | OOD IR = {np.mean(ood_irs):.4f} (Retention = {quantity_results[q_name]['retention']:.3f})")

    results_payload["training_quantity_ablation"] = quantity_results

    # =========================================================================
    # STEP 6: TEMPORAL FEATURE ABLATION (EXPERIMENT 10)
    # =========================================================================
    print("\n--- STEP 6: Temporal Feature Ablation (On Diversified Training) ---")
    ablation_builders = {
        "baseline_encoder": lambda nb: TemporalObservationEncoder(nb, include_absolute_time=True),
        "plus_freq_movement": lambda nb: TemporalObservationEncoder(nb, include_absolute_time=True, include_freq_movement=True),
        "plus_det_rate": lambda nb: TemporalObservationEncoder(nb, include_absolute_time=True, include_det_rate=True),
        "plus_persistence": lambda nb: TemporalObservationEncoder(nb, include_absolute_time=True, include_persistence=True),
        "plus_time_between_det": lambda nb: TemporalObservationEncoder(nb, include_absolute_time=True, include_time_between_detections=True),
        "plus_consistency": lambda nb: TemporalObservationEncoder(nb, include_absolute_time=True, include_consistency=True),
        "all_temporal_features": lambda nb: TemporalObservationEncoder.create_encoder_b(nb),
    }

    feature_ablation_results: dict[str, Any] = {}
    for feat_name, builder in ablation_builders.items():
        feat_models = []
        for t_seed in TRAIN_SEEDS:
            rng = np.random.default_rng(t_seed)
            dummy_sc, _ = generate_training_scenario(0, rng)
            env = build_environment(dummy_sc)
            enc = builder(len(env.bands_hz))
            sched = DDQNScheduler(bands_hz=env.bands_hz, encoder=enc, seed=t_seed)
            sched.train()
            for ep in range(6):  # 6 episodes
                ep_sc, _ = generate_training_scenario(ep, rng)
                env_ep = build_environment(ep_sc)
                saved_eps = sched.epsilon
                env_ep.scheduler = sched
                env_ep.reset(seed=t_seed + ep * 500)
                sched.epsilon = saved_eps
                env_ep.run(steps=300)
            feat_models.append(sched)

        id_irs = [float(np.mean([evaluate_scheduler(sc_A, m, seed=s, steps=300)["interception_ratio"] for m in feat_models])) for s in EVAL_SEEDS]
        ood_irs = [float(np.mean([evaluate_scheduler(sc_E, m, seed=s, steps=300)["interception_ratio"] for m in feat_models])) for s in EVAL_SEEDS]

        feature_ablation_results[feat_name] = {
            "id_ir_mean": float(np.mean(id_irs)),
            "ood_ir_mean": float(np.mean(ood_irs)),
            "retention": float(np.mean(ood_irs)) / max(float(np.mean(id_irs)), 1e-6),
        }
        print(f"   {feat_name:<25}: ID IR = {np.mean(id_irs):.4f} | OOD IR = {np.mean(ood_irs):.4f} | Retention = {feature_ablation_results[feat_name]['retention']:.3f}")

    results_payload["feature_ablation"] = feature_ablation_results

    # =========================================================================
    # STEP 7: PERIODIC BURST EVALUATION (EXPERIMENT 17)
    # =========================================================================
    print("\n--- STEP 7: Periodic Burst In-Depth Check ---")
    p_sc = scenarios["periodic_time"]
    ca_p = evaluate_baseline(runner, p_sc, "context_aware", EVAL_SEEDS, steps=300)

    # 1. V3.0 DDQN (trained on periodic_time)
    v3_p_irs = [float(np.mean([evaluate_scheduler(p_sc, m, seed=s, steps=300)["interception_ratio"] for m in sub_models])) for s in EVAL_SEEDS]

    # 2. V3.1 Temporal-feature DDQN (Encoder B trained on periodic_time)
    b_p_models = []
    for t_seed in TRAIN_SEEDS:
        env = build_environment(copy.deepcopy(p_sc))
        enc = TemporalObservationEncoder.create_encoder_b(len(env.bands_hz))
        sched = DDQNScheduler(bands_hz=env.bands_hz, encoder=enc, seed=t_seed)
        sched.train()
        for ep in range(4):
            saved_eps = sched.epsilon
            env.reset(seed=t_seed + ep * 1000)
            sched.epsilon = saved_eps
            env.run(steps=300)
        b_p_models.append(sched)
    b_p_irs = [float(np.mean([evaluate_scheduler(p_sc, m, seed=s, steps=300)["interception_ratio"] for m in b_p_models])) for s in EVAL_SEEDS]

    # 3. V3.1 Periodic-feature DDQN (Encoder D Fourier)
    d_p_models = []
    for t_seed in TRAIN_SEEDS:
        env = build_environment(copy.deepcopy(p_sc))
        enc = TemporalObservationEncoder.create_encoder_d(len(env.bands_hz))
        sched = DDQNScheduler(bands_hz=env.bands_hz, encoder=enc, seed=t_seed)
        sched.train()
        for ep in range(4):
            saved_eps = sched.epsilon
            env.reset(seed=t_seed + ep * 1000)
            sched.epsilon = saved_eps
            env.run(steps=300)
        d_p_models.append(sched)
    d_p_irs = [float(np.mean([evaluate_scheduler(p_sc, m, seed=s, steps=300)["interception_ratio"] for m in d_p_models])) for s in EVAL_SEEDS]

    # 4. Diversified DDQN
    div_p_irs = [float(np.mean([evaluate_scheduler(p_sc, m, seed=s, steps=300)["interception_ratio"] for m in diversified_models])) for s in EVAL_SEEDS]

    periodic_burst_comparison = {
        "context_aware": {"ir_mean": ca_p["ir_mean"], "ir_std": ca_p["ir_std"]},
        "v3_0_ddqn": {"ir_mean": float(np.mean(v3_p_irs)), "ir_std": float(np.std(v3_p_irs))},
        "v3_1_temporal_ddqn": {"ir_mean": float(np.mean(b_p_irs)), "ir_std": float(np.std(b_p_irs))},
        "v3_1_fourier_ddqn": {"ir_mean": float(np.mean(d_p_irs)), "ir_std": float(np.std(d_p_irs))},
        "v3_1_diversified_ddqn": {"ir_mean": float(np.mean(div_p_irs)), "ir_std": float(np.std(div_p_irs))},
    }
    print("   Periodic Burst Results:")
    for k, v in periodic_burst_comparison.items():
        print(f"      {k:<25}: IR = {v['ir_mean']:.4f} ± {v['ir_std']:.4f}")
    results_payload["periodic_burst_comparison"] = periodic_burst_comparison

    # =========================================================================
    # STEP 8: STATISTICAL TESTS & BEHAVIORAL TRAJECTORIES (EXPERIMENTS 21 & 22)
    # =========================================================================
    print("\n--- STEP 8: Statistical Analysis & Behavioral Extraction ---")

    # Statistical test across all 6 generalization conditions: V3.1 DDQN vs Context-Aware
    all_v31_irs = []
    all_ca_irs = []
    for cond_name, cond_sc in conditions.items():
        # 10 seeds
        for e_s in EVAL_SEEDS:
            # CA
            ca_val = runner.run_single(cond_sc, "context_aware", seed=e_s, steps=300)["interception_ratio"]
            all_ca_irs.append(ca_val)
            # V3.1
            v31_val = float(np.mean([evaluate_scheduler(cond_sc, m, seed=e_s, steps=300)["interception_ratio"] for m in diversified_models]))
            all_v31_irs.append(v31_val)

    diffs = np.array(all_v31_irs, dtype=np.float64) - np.array(all_ca_irs, dtype=np.float64)
    n_samples = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1)) if n_samples > 1 else 0.0
    se_d = math.sqrt(var_d / n_samples) if var_d > 0 else 1e-12
    t_stat = mean_d / se_d
    z = abs(t_stat)
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    p_val = max(min(p_val, 1.0), 0.0)
    t_crit = 2.000 if n_samples >= 30 else 2.262
    ci_low = mean_d - t_crit * se_d
    ci_high = mean_d + t_crit * se_d

    stat_results = {
        "overall_v31_mean": float(np.mean(all_v31_irs)),
        "overall_ca_mean": float(np.mean(all_ca_irs)),
        "mean_diff": float(np.mean(diffs)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "ci_95": [float(ci_low), float(ci_high)],
        "significant_05": bool(p_val < 0.05),
    }
    print(f"   Overall Comparison: V3.1 Mean IR = {stat_results['overall_v31_mean']:.4f} vs CA Mean IR = {stat_results['overall_ca_mean']:.4f}")
    print(f"   Mean Diff = {stat_results['mean_diff']:.4f} | t = {t_stat:.4f} | p = {p_val:.4e} | 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    results_payload["statistical_analysis"] = stat_results

    # Behavioral trajectories (30 steps each)
    best_model = diversified_models[0]
    best_model.eval()

    def capture_trajectory(sc: dict[str, Any]) -> list[dict[str, Any]]:
        env = build_environment(copy.deepcopy(sc))
        num_b = len(env.bands_hz)
        class SafeWrapper:
            def __init__(self, inner: Any, max_b: int) -> None:
                self._inner = inner
                self._max_b = max_b
            def select_action(self, obs: SchedulerObservation) -> ScanAction:
                act = self._inner.select_action(obs)
                b = act.frequency_bin if isinstance(act, ScanAction) else int(act)
                return ScanAction(frequency_bin=min(max(b, 0), self._max_b - 1))
            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)
        env.scheduler = SafeWrapper(best_model, num_b)
        steps = env.run(steps=30)
        traj = []
        for s in steps:
            obs = s.observation
            b_sel = obs.current_frequency_bin if obs.current_frequency_bin is not None else 0
            traj.append({
                "step": obs.timestamp,
                "selected_bin": b_sel,
                "center_freq_mhz": bin_to_center_hz(b_sel) / 1e6,
                "detected": bool(obs.last_detection),
                "reward": s.reward,
            })
        return traj

    trajectories = {
        "deterministic_hopping": capture_trajectory(sc_A),
        "ood_hopping": capture_trajectory(sc_B),
        "periodic_burst": capture_trajectory(scenarios["periodic_time"]),
        "random_hopping": capture_trajectory(scenarios["random_hopping"]),
    }
    results_payload["behavioral_trajectories"] = trajectories

    # Save complete results payload
    output_path = Path("scratch/validation/v3_1_study_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n✅ Successfully saved full study results to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
