from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.environment.builder import build_environment
from rf_environment.environment.scenario import load_scenario
from rf_environment.experiments.runner import BenchmarkRunner
from rf_environment.scheduler.context_aware import ContextAwareScheduler
from rf_environment.scheduler.hybrid.arbitrator import ArbitrationMode, HybridMetaArbitrator
from rf_environment.scheduler.hybrid.hybrid_scheduler import HybridScheduler
from rf_environment.scheduler.rl.ddqn_scheduler import DDQNScheduler
from rf_environment.scheduler.rl.encoder import ObservationEncoder
from rf_environment.scheduler.rl.temporal_encoder import TemporalObservationEncoder
from rf_environment.util.seeding import derive_seed


TRAIN_SEEDS = [42, 123, 456, 789]
EVAL_SEEDS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

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


def evaluate_scheduler_run(
    env_scenario: dict[str, Any],
    scheduler: Any,
    seed: int,
    steps: int = 300,
) -> dict[str, Any]:
    scenario_copy = copy.deepcopy(env_scenario)
    scenario_copy["simulation"]["seed"] = seed
    scenario_copy["simulation"]["total_time_steps"] = steps

    env = build_environment(scenario_copy)
    scheduler.eval()
    if hasattr(scheduler, "epsilon"):
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

        def observe(self, *args: Any, **kwargs: Any) -> None:
            if hasattr(self._inner, "observe"):
                self._inner.observe(*args, **kwargs)

        def reset(self) -> None:
            if hasattr(self._inner, "reset"):
                self._inner.reset()

        def reseed(self, s: int) -> None:
            if hasattr(self._inner, "reseed"):
                self._inner.reseed(s)

        def get_diagnostics(self) -> dict[str, Any]:
            if hasattr(self._inner, "get_diagnostics"):
                return self._inner.get_diagnostics()
            return {}

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    wrapped = SafeSchedulerWrapper(scheduler, num_bins)
    env.scheduler = wrapped
    wrapped.reset()

    res = env.run(steps=steps)
    snap = env.metrics.snapshot(env.clock.time_step).to_dict()

    hits = sum(1 for r in res if r.observation.last_detection)
    det_rate = float(hits) / float(len(res)) if res else 0.0

    diag = wrapped.get_diagnostics()

    return {
        "interception_ratio": float(snap.get("interception_ratio", 0.0)),
        "opportunity_coverage": float(snap.get("opportunity_coverage", 0.0)),
        "detection_given_coverage": float(snap.get("detection_given_coverage", 0.0)),
        "detection_rate": det_rate,
        "diagnostics": diag,
        "step_results": res,
    }


def evaluate_baseline_runner(
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
    t_start = time.time()
    print("=" * 80)
    print("🚀 SIH26055 — V4.0 HYBRID SCHEDULER PERFORMANCE BENCHMARK")
    print("=" * 80)

    runner = BenchmarkRunner()
    results_payload: dict[str, Any] = {}

    # Define the 8 evaluation scenarios:
    # 1. Seen Structure: [5, 15, 25] (dwell=3)
    sc_A = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]], "mode": "sequential", "dwell_steps": 3},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # 2. Unseen Permutation: [25, 5, 15] (dwell=3)
    sc_B = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [25, 5, 15]], "mode": "sequential", "dwell_steps": 3},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # 3. Unseen Phase: [15, 25, 5] (dwell=3)
    sc_C = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [15, 25, 5]], "mode": "sequential", "dwell_steps": 3},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # 4. Unseen Dwell: [5, 15, 25] (dwell=1)
    sc_D = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]], "mode": "sequential", "dwell_steps": 1},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # 5. Unseen Subset: [10, 20, 27] (dwell=3)
    sc_E = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [10, 20, 27]], "mode": "sequential", "dwell_steps": 3},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # 6. Mixed Shift: [10, 27, 20] (dwell=5)
    sc_F = make_scenario([{
        "id": "EM", "type": "radar",
        "frequency_behavior": {"type": "hopping", "frequencies_hz": [bin_to_center_hz(b) for b in [10, 27, 20]], "mode": "sequential", "dwell_steps": 5},
        "time_behavior": {"type": "continuous"}, "power_dbm": -18, "bandwidth_hz": 2_000_000
    }])

    # 7. Random Hopping
    sc_G = load_scenario("rf_environment/scenarios/random_hopping.yaml")

    # 8. Periodic Burst: fixed frequency emitter with periodic on/off (from periodic_time.yaml)
    sc_H = make_scenario([{
        "id": "E_PERIODIC",
        "type": "communication",
        "frequency_behavior": {"type": "fixed", "frequency_hz": 400_000_000},
        "time_behavior": {"type": "periodic", "on_duration": 5, "off_duration": 15, "phase": 0},
        "power_dbm": -22,
        "bandwidth_hz": 2_000_000,
    }])

    scenarios = {
        "1_Seen_Structure": sc_A,
        "2_Unseen_Permutation": sc_B,
        "3_Unseen_Phase": sc_C,
        "4_Unseen_Dwell": sc_D,
        "5_Unseen_Subset": sc_E,
        "6_Mixed_Shift": sc_F,
        "7_Random_Hopping": sc_G,
        "8_Periodic_Burst": sc_H,
    }

    # =========================================================================
    # STEP 1: PRE-TRAIN BASELINE MODELS ON SEEN STRUCTURE
    # =========================================================================
    print("\n--- STEP 1: Training DDQN Models on Seen Deterministic Hopping ---")
    v3_0_models = []
    v3_1_models = []

    for t_seed in TRAIN_SEEDS:
        # V3.0 (Encoder A)
        env_v3_0 = build_environment(copy.deepcopy(sc_A))
        enc_a = ObservationEncoder(num_bins=len(env_v3_0.bands_hz))
        m_v3_0 = DDQNScheduler(bands_hz=env_v3_0.bands_hz, encoder=enc_a, seed=t_seed)
        m_v3_0.train()
        env_v3_0.scheduler = m_v3_0

        # V3.1 (Encoder B - Temporal)
        env_v3_1 = build_environment(copy.deepcopy(sc_A))
        enc_b = TemporalObservationEncoder.create_encoder_b(num_bins=len(env_v3_1.bands_hz))
        m_v3_1 = DDQNScheduler(bands_hz=env_v3_1.bands_hz, encoder=enc_b, seed=t_seed)
        m_v3_1.train()
        env_v3_1.scheduler = m_v3_1

        for ep in range(4):
            ep_seed = t_seed + ep * 1000
            # V3.0 train episode
            s_eps0 = m_v3_0.epsilon
            env_v3_0.reset(seed=ep_seed)
            m_v3_0.epsilon = s_eps0
            env_v3_0.run(steps=300)

            # V3.1 train episode
            s_eps1 = m_v3_1.epsilon
            env_v3_1.reset(seed=ep_seed)
            m_v3_1.epsilon = s_eps1
            env_v3_1.run(steps=300)

        v3_0_models.append(m_v3_0)
        v3_1_models.append(m_v3_1)

    print(f"   Trained {len(v3_0_models)} V3.0 models and {len(v3_1_models)} V3.1 models.")

    # Pick the representative model from the ensemble (seed 42)
    rep_v3_0 = v3_0_models[0]
    rep_v3_1 = v3_1_models[0]

    # =========================================================================
    # STEP 2: BENCHMARK ACROSS 8 SCENARIOS & 10 EVAL SEEDS
    # =========================================================================
    print("\n--- STEP 2: Running 8-Scenario Benchmark ---")

    benchmark_summary: dict[str, Any] = {}

    for sc_name, sc_cfg in scenarios.items():
        print(f"\nEvaluating: {sc_name}...")

        # 1. Context-Aware
        ca_res = evaluate_baseline_runner(runner, sc_cfg, "context_aware", EVAL_SEEDS, steps=300)

        # 2. V3.0 DDQN (Encoder A)
        v3_0_irs, v3_0_covs, v3_0_dets = [], [], []
        for e_s in EVAL_SEEDS:
            # Evaluate across trained ensemble models
            s_irs = []
            for m in v3_0_models:
                m_copy = copy.deepcopy(m)
                ev = evaluate_scheduler_run(sc_cfg, m_copy, seed=e_s, steps=300)
                s_irs.append(ev["interception_ratio"])
            v3_0_irs.append(float(np.mean(s_irs)))

        # 3. V3.1 DDQN (Encoder B)
        v3_1_irs, v3_1_covs, v3_1_dets = [], [], []
        for e_s in EVAL_SEEDS:
            s_irs = []
            for m in v3_1_models:
                m_copy = copy.deepcopy(m)
                ev = evaluate_scheduler_run(sc_cfg, m_copy, seed=e_s, steps=300)
                s_irs.append(ev["interception_ratio"])
            v3_1_irs.append(float(np.mean(s_irs)))

        # 4. V4.0 Hybrid Scheduler
        v4_irs, v4_covs, v4_dets, v4_diags = [], [], [], []
        adaptation_steps_list = []

        for e_s in EVAL_SEEDS:
            seed_v4_irs, seed_v4_covs, seed_v4_dets = [], [], []
            seed_adapt_steps = []

            for m in v3_1_models:
                env_tmp = build_environment(copy.deepcopy(sc_cfg))
                bands = env_tmp.bands_hz
                trained_ddqn = copy.deepcopy(m)
                trained_ddqn.bands_hz = bands
                trained_ddqn.eval()
                trained_ddqn.epsilon = 0.0

                hybrid = HybridScheduler(
                    bands_hz=bands,
                    ddqn=trained_ddqn,
                    seed=derive_seed(e_s, "hybrid"),
                )
                hybrid.eval()

                ev = evaluate_scheduler_run(sc_cfg, hybrid, seed=e_s, steps=300)
                seed_v4_irs.append(ev["interception_ratio"])
                seed_v4_covs.append(ev["opportunity_coverage"])
                seed_v4_dets.append(ev["detection_rate"])
                v4_diags.append(ev["diagnostics"])

                step_res = ev["step_results"]
                first_hit_step = next((idx for idx, r in enumerate(step_res) if r.observation.last_detection), 300)
                seed_adapt_steps.append(first_hit_step)

            v4_irs.append(float(np.mean(seed_v4_irs)))
            v4_covs.append(float(np.mean(seed_v4_covs)))
            v4_dets.append(float(np.mean(seed_v4_dets)))
            adaptation_steps_list.append(float(np.mean(seed_adapt_steps)))

        # Compute aggregate diagnostics for V4 Hybrid
        avg_pct_exploit = float(np.mean([d.get("pct_ddqn_exploit", 0.0) for d in v4_diags]))
        avg_pct_ca_adapt = float(np.mean([d.get("pct_ca_adapt", 0.0) for d in v4_diags]))
        avg_pct_blended = float(np.mean([d.get("pct_blended", 0.0) for d in v4_diags]))
        avg_pct_explore = float(np.mean([d.get("pct_explore", 0.0) for d in v4_diags]))
        avg_w_ddqn = float(np.mean([d.get("avg_ddqn_weight", 0.0) for d in v4_diags]))
        avg_w_ca = float(np.mean([d.get("avg_ca_weight", 0.0) for d in v4_diags]))
        avg_adapt_steps = float(np.mean(adaptation_steps_list))

        sc_summary = {
            "Context_Aware": {
                "ir_mean": ca_res["ir_mean"],
                "ir_std": ca_res["ir_std"],
                "cov_mean": ca_res["cov_mean"],
                "det_rate_mean": ca_res["det_rate_mean"],
            },
            "V3_0_DDQN": {
                "ir_mean": float(np.mean(v3_0_irs)),
                "ir_std": float(np.std(v3_0_irs)),
            },
            "V3_1_DDQN": {
                "ir_mean": float(np.mean(v3_1_irs)),
                "ir_std": float(np.std(v3_1_irs)),
            },
            "V4_0_Hybrid": {
                "ir_mean": float(np.mean(v4_irs)),
                "ir_std": float(np.std(v4_irs)),
                "cov_mean": float(np.mean(v4_covs)),
                "det_rate_mean": float(np.mean(v4_dets)),
                "pct_ddqn_exploit": avg_pct_exploit,
                "pct_ca_adapt": avg_pct_ca_adapt,
                "pct_blended": avg_pct_blended,
                "pct_explore": avg_pct_explore,
                "avg_w_ddqn": avg_w_ddqn,
                "avg_w_ca": avg_w_ca,
                "avg_adaptation_steps": avg_adapt_steps,
            },
        }

        benchmark_summary[sc_name] = sc_summary

        print(
            f"   Results for {sc_name}:\n"
            f"      CA:        IR = {ca_res['ir_mean']*100:.1f}%\n"
            f"      V3.0 DDQN: IR = {np.mean(v3_0_irs)*100:.1f}%\n"
            f"      V3.1 DDQN: IR = {np.mean(v3_1_irs)*100:.1f}%\n"
            f"      V4.0 Hyb:  IR = {np.mean(v4_irs)*100:.1f}% "
            f"[Exploit: {avg_pct_exploit:.0f}%, Adapt: {avg_pct_ca_adapt:.0f}%, Blend: {avg_pct_blended:.0f}%, Exp: {avg_pct_explore:.0f}%]"
        )

    # =========================================================================
    # STEP 3: GENERALIZATION & OOD RETENTION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY & OOD GENERALIZATION TABLE")
    print("=" * 80)

    header = f"{'Scenario':<24} | {'CA IR':<9} | {'V3.0 IR':<9} | {'V3.1 IR':<9} | {'V4.0 Hyb':<9} | {'V4 Mode Breakdown (% Exp / Ad / Bl / Disc)':<40}"
    print(header)
    print("-" * len(header))

    seen_v3_0 = benchmark_summary["1_Seen_Structure"]["V3_0_DDQN"]["ir_mean"]
    seen_v3_1 = benchmark_summary["1_Seen_Structure"]["V3_1_DDQN"]["ir_mean"]
    seen_v4 = benchmark_summary["1_Seen_Structure"]["V4_0_Hybrid"]["ir_mean"]

    ood_scenarios = [
        "2_Unseen_Permutation",
        "3_Unseen_Phase",
        "4_Unseen_Dwell",
        "5_Unseen_Subset",
        "6_Mixed_Shift",
        "7_Random_Hopping",
        "8_Periodic_Burst",
    ]

    retention_table: dict[str, Any] = {}

    for sc_name, sc_data in benchmark_summary.items():
        ca_ir = sc_data["Context_Aware"]["ir_mean"] * 100
        v30_ir = sc_data["V3_0_DDQN"]["ir_mean"] * 100
        v31_ir = sc_data["V3_1_DDQN"]["ir_mean"] * 100
        v4_ir = sc_data["V4_0_Hybrid"]["ir_mean"] * 100
        diag_str = (
            f"{sc_data['V4_0_Hybrid']['pct_ddqn_exploit']:.0f}% / "
            f"{sc_data['V4_0_Hybrid']['pct_ca_adapt']:.0f}% / "
            f"{sc_data['V4_0_Hybrid']['pct_blended']:.0f}% / "
            f"{sc_data['V4_0_Hybrid']['pct_explore']:.0f}%"
        )
        print(f"{sc_name:<24} | {ca_ir:>8.1f}% | {v30_ir:>8.1f}% | {v31_ir:>8.1f}% | {v4_ir:>8.1f}% | {diag_str:<40}")

    print("\n" + "=" * 80)
    print("🛡️ OOD RETENTION RATIO (IR_OOD / IR_Seen)")
    print("=" * 80)
    print(f"{'Scenario':<24} | {'V3.0 Retention':<16} | {'V3.1 Retention':<16} | {'V4.0 Retention':<16}")
    print("-" * 80)

    for ood_name in ood_scenarios:
        r_v30 = benchmark_summary[ood_name]["V3_0_DDQN"]["ir_mean"] / max(seen_v3_0, 1e-4) * 100
        r_v31 = benchmark_summary[ood_name]["V3_1_DDQN"]["ir_mean"] / max(seen_v3_1, 1e-4) * 100
        r_v4 = benchmark_summary[ood_name]["V4_0_Hybrid"]["ir_mean"] / max(seen_v4, 1e-4) * 100
        retention_table[ood_name] = {
            "v3_0_retention": r_v30,
            "v3_1_retention": r_v31,
            "v4_0_retention": r_v4,
        }
        print(f"{ood_name:<24} | {r_v30:>15.1f}% | {r_v31:>15.1f}% | {r_v4:>15.1f}%")

    results_payload["benchmark_summary"] = benchmark_summary
    results_payload["ood_retention"] = retention_table
    results_payload["total_runtime_seconds"] = time.time() - t_start

    out_file = Path("scratch/validation/v4_benchmark_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n✅ Benchmark results saved to {out_file} in {results_payload['total_runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
