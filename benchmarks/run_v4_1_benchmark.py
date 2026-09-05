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
from rf_environment.scheduler.hybrid.hybrid_v41 import LSTMHybridScheduler
from rf_environment.scheduler.rl.ddqn_scheduler import DDQNScheduler
from rf_environment.scheduler.rl.encoder import ObservationEncoder
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
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
    print("🚀 SIH26055 — V4.1 LSTM-HYBRID SCHEDULER PERFORMANCE BENCHMARK")
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

    # 8. Periodic Burst: fixed frequency emitter with periodic on/off
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
    # STEP 1: PRE-TRAIN MODELS ON SEEN STRUCTURE
    # =========================================================================
    print("\n--- STEP 1: Pre-training DDQN, V3.1, and LSTM-DDQN on Seen Hopping ---")
    v3_0_models = []
    v3_1_models = []
    v4_1_lstm_models = []

    for t_seed in TRAIN_SEEDS:
        # V3.0 (Feed-forward Encoder A)
        env_v3_0 = build_environment(copy.deepcopy(sc_A))
        enc_a = ObservationEncoder(num_bins=len(env_v3_0.bands_hz))
        m_v3_0 = DDQNScheduler(bands_hz=env_v3_0.bands_hz, encoder=enc_a, seed=t_seed)
        m_v3_0.train()
        env_v3_0.scheduler = m_v3_0

        # V3.1 (Feed-forward Encoder B - Temporal)
        env_v3_1 = build_environment(copy.deepcopy(sc_A))
        enc_b = TemporalObservationEncoder.create_encoder_b(num_bins=len(env_v3_1.bands_hz))
        m_v3_1 = DDQNScheduler(bands_hz=env_v3_1.bands_hz, encoder=enc_b, seed=t_seed)
        m_v3_1.train()
        env_v3_1.scheduler = m_v3_1

        # V4.1 (Recurrent LSTM-DDQN - Encoder B)
        env_v4_1 = build_environment(copy.deepcopy(sc_A))
        enc_lstm = TemporalObservationEncoder.create_encoder_b(num_bins=len(env_v4_1.bands_hz))
        m_lstm = LSTMDDQNScheduler(
            bands_hz=env_v4_1.bands_hz,
            encoder=enc_lstm,
            hidden_dim=64,
            dense_dim=64,
            sequence_length=10,
            learning_rate=0.001,
            seed=t_seed,
        )
        m_lstm.train()
        env_v4_1.scheduler = m_lstm

        # 5 training episodes per model
        for ep in range(5):
            ep_seed = t_seed + ep * 1000
            # Train V3.0
            s_eps0 = m_v3_0.epsilon
            env_v3_0.reset(seed=ep_seed)
            m_v3_0.epsilon = s_eps0
            env_v3_0.run(steps=300)

            # Train V3.1
            s_eps1 = m_v3_1.epsilon
            env_v3_1.reset(seed=ep_seed)
            m_v3_1.epsilon = s_eps1
            env_v3_1.run(steps=300)

            # Train LSTM-DDQN
            s_eps_l = m_lstm.epsilon
            env_v4_1.reset(seed=ep_seed)
            m_lstm.epsilon = s_eps_l
            env_v4_1.run(steps=300)

        # Persist trained model checkpoint
        models_dir = Path("models/v4_1")
        models_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = models_dir / f"lstm_ddqn_seed{t_seed}.npz"
        m_lstm.save_checkpoint(
            ckpt_path,
            metadata={
                "training_seed": t_seed,
                "episodes": 5,
                "scenario": "1_Seen_Structure",
                "active_bins": [5, 15, 25],
                "dwell_steps": 3,
            },
        )
        print(f"   [Checkpoint Saved] Seed {t_seed:3d} -> {ckpt_path} (SHA256: {m_lstm.checkpoint_sha256[:16]}...)")

        if t_seed == 42:
            prod_path = models_dir / "production_checkpoint.npz"
            m_lstm.save_checkpoint(
                prod_path,
                metadata={
                    "training_seed": t_seed,
                    "episodes": 5,
                    "scenario": "1_Seen_Structure",
                    "active_bins": [5, 15, 25],
                    "dwell_steps": 3,
                    "role": "production_reference",
                },
            )
            print(f"   [Production Checkpoint Saved] -> {prod_path} (SHA256: {m_lstm.checkpoint_sha256})")

        v3_0_models.append(m_v3_0)
        v3_1_models.append(m_v3_1)
        v4_1_lstm_models.append(m_lstm)

    print(f"   Pre-training complete: {len(v3_0_models)} V3.0, {len(v3_1_models)} V3.1, {len(v4_1_lstm_models)} LSTM-DDQN models.")

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
        v3_0_irs = []
        for e_s in EVAL_SEEDS:
            s_irs = []
            for m in v3_0_models:
                m_copy = copy.deepcopy(m)
                ev = evaluate_scheduler_run(sc_cfg, m_copy, seed=e_s, steps=300)
                s_irs.append(ev["interception_ratio"])
            v3_0_irs.append(float(np.mean(s_irs)))

        # 3. V3.1 DDQN (Encoder B)
        v3_1_irs = []
        for e_s in EVAL_SEEDS:
            s_irs = []
            for m in v3_1_models:
                m_copy = copy.deepcopy(m)
                ev = evaluate_scheduler_run(sc_cfg, m_copy, seed=e_s, steps=300)
                s_irs.append(ev["interception_ratio"])
            v3_1_irs.append(float(np.mean(s_irs)))

        # 4. V4.0 Hybrid Scheduler
        v4_0_irs, v4_0_covs, v4_0_dets = [], [], []
        v4_0_diags = []
        for e_s in EVAL_SEEDS:
            s_irs, s_covs, s_dets = [], [], []
            for m in v3_1_models:
                env_tmp = build_environment(copy.deepcopy(sc_cfg))
                bands = env_tmp.bands_hz
                trained_ddqn = copy.deepcopy(m)
                trained_ddqn.bands_hz = bands
                trained_ddqn.eval()
                trained_ddqn.epsilon = 0.0

                hybrid_v40 = HybridScheduler(
                    bands_hz=bands,
                    ddqn=trained_ddqn,
                    seed=derive_seed(e_s, "hybrid_v40"),
                )
                hybrid_v40.eval()

                ev = evaluate_scheduler_run(sc_cfg, hybrid_v40, seed=e_s, steps=300)
                s_irs.append(ev["interception_ratio"])
                s_covs.append(ev["opportunity_coverage"])
                s_dets.append(ev["detection_rate"])
                v4_0_diags.append(ev["diagnostics"])

            v4_0_irs.append(float(np.mean(s_irs)))
            v4_0_covs.append(float(np.mean(s_covs)))
            v4_0_dets.append(float(np.mean(s_dets)))

        # 5. Standalone LSTM-DDQN (DRQN)
        lstm_ddqn_irs = []
        for e_s in EVAL_SEEDS:
            s_irs = []
            for m in v4_1_lstm_models:
                m_copy = copy.deepcopy(m)
                ev = evaluate_scheduler_run(sc_cfg, m_copy, seed=e_s, steps=300)
                s_irs.append(ev["interception_ratio"])
            lstm_ddqn_irs.append(float(np.mean(s_irs)))

        # 6. V4.1 LSTM-Hybrid Scheduler
        v4_1_irs, v4_1_covs, v4_1_dets = [], [], []
        v4_1_diags = []
        v4_1_adapt_steps = []

        for e_s in EVAL_SEEDS:
            s_irs, s_covs, s_dets = [], [], []
            seed_adapt_steps = []
            for m in v4_1_lstm_models:
                env_tmp = build_environment(copy.deepcopy(sc_cfg))
                bands = env_tmp.bands_hz
                trained_lstm = copy.deepcopy(m)
                trained_lstm.bands_hz = bands
                trained_lstm.eval()
                trained_lstm.epsilon = 0.0

                hybrid_v41 = LSTMHybridScheduler(
                    bands_hz=bands,
                    lstm_ddqn=trained_lstm,
                    seed=derive_seed(e_s, "hybrid_v41"),
                )
                hybrid_v41.eval()

                ev = evaluate_scheduler_run(sc_cfg, hybrid_v41, seed=e_s, steps=300)
                s_irs.append(ev["interception_ratio"])
                s_covs.append(ev["opportunity_coverage"])
                s_dets.append(ev["detection_rate"])
                v4_1_diags.append(ev["diagnostics"])

                step_res = ev["step_results"]
                first_hit = next((idx for idx, r in enumerate(step_res) if r.observation.last_detection), 300)
                seed_adapt_steps.append(first_hit)

            v4_1_irs.append(float(np.mean(s_irs)))
            v4_1_covs.append(float(np.mean(s_covs)))
            v4_1_dets.append(float(np.mean(s_dets)))
            v4_1_adapt_steps.append(float(np.mean(seed_adapt_steps)))

        # Aggregate diagnostics for V4.1 Hybrid
        avg_v41_exploit = float(np.mean([d.get("pct_ddqn_exploit", 0.0) for d in v4_1_diags]))
        avg_v41_ca_adapt = float(np.mean([d.get("pct_ca_adapt", 0.0) for d in v4_1_diags]))
        avg_v41_blended = float(np.mean([d.get("pct_blended", 0.0) for d in v4_1_diags]))
        avg_v41_explore = float(np.mean([d.get("pct_explore", 0.0) for d in v4_1_diags]))
        avg_v41_w_ddqn = float(np.mean([d.get("avg_ddqn_weight", 0.0) for d in v4_1_diags]))
        avg_v41_w_ca = float(np.mean([d.get("avg_ca_weight", 0.0) for d in v4_1_diags]))
        avg_v41_adapt = float(np.mean(v4_1_adapt_steps))

        # Aggregate diagnostics for V4.0 Hybrid
        avg_v40_exploit = float(np.mean([d.get("pct_ddqn_exploit", 0.0) for d in v4_0_diags]))
        avg_v40_ca_adapt = float(np.mean([d.get("pct_ca_adapt", 0.0) for d in v4_0_diags]))
        avg_v40_blended = float(np.mean([d.get("pct_blended", 0.0) for d in v4_0_diags]))
        avg_v40_explore = float(np.mean([d.get("pct_explore", 0.0) for d in v4_0_diags]))

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
                "ir_mean": float(np.mean(v4_0_irs)),
                "ir_std": float(np.std(v4_0_irs)),
                "cov_mean": float(np.mean(v4_0_covs)),
                "det_rate_mean": float(np.mean(v4_0_dets)),
                "pct_ddqn_exploit": avg_v40_exploit,
                "pct_ca_adapt": avg_v40_ca_adapt,
                "pct_blended": avg_v40_blended,
                "pct_explore": avg_v40_explore,
            },
            "V4_1_LSTM_DDQN": {
                "ir_mean": float(np.mean(lstm_ddqn_irs)),
                "ir_std": float(np.std(lstm_ddqn_irs)),
            },
            "V4_1_LSTM_Hybrid": {
                "ir_mean": float(np.mean(v4_1_irs)),
                "ir_std": float(np.std(v4_1_irs)),
                "cov_mean": float(np.mean(v4_1_covs)),
                "det_rate_mean": float(np.mean(v4_1_dets)),
                "pct_ddqn_exploit": avg_v41_exploit,
                "pct_ca_adapt": avg_v41_ca_adapt,
                "pct_blended": avg_v41_blended,
                "pct_explore": avg_v41_explore,
                "avg_w_ddqn": avg_v41_w_ddqn,
                "avg_w_ca": avg_v41_w_ca,
                "avg_adaptation_steps": avg_v41_adapt,
            },
        }

        benchmark_summary[sc_name] = sc_summary

        print(
            f"   Results for {sc_name}:\n"
            f"      CA:               IR = {ca_res['ir_mean']*100:.1f}%\n"
            f"      V3.0 DDQN:        IR = {np.mean(v3_0_irs)*100:.1f}%\n"
            f"      V3.1 DDQN:        IR = {np.mean(v3_1_irs)*100:.1f}%\n"
            f"      V4.0 Hybrid:      IR = {np.mean(v4_0_irs)*100:.1f}%\n"
            f"      V4.1 LSTM-DDQN:   IR = {np.mean(lstm_ddqn_irs)*100:.1f}%\n"
            f"      V4.1 LSTM-Hybrid: IR = {np.mean(v4_1_irs)*100:.1f}% "
            f"[Exploit: {avg_v41_exploit:.0f}%, Adapt: {avg_v41_ca_adapt:.0f}%, Blend: {avg_v41_blended:.0f}%, Exp: {avg_v41_explore:.0f}%]"
        )

    # =========================================================================
    # STEP 3: GENERALIZATION & OOD RETENTION ANALYSIS
    # =========================================================================
    print("\n" + "=" * 105)
    print("📊 BENCHMARK SUMMARY & OOD GENERALIZATION TABLE (V4.1 LSTM-HYBRID)")
    print("=" * 105)

    header = f"{'Scenario':<22} | {'CA IR':<7} | {'V3.0 IR':<8} | {'V3.1 IR':<8} | {'V4.0 Hyb':<8} | {'V4.1 LSTM':<9} | {'V4.1 Hyb':<8} | {'V4.1 Modes (% Exp/Ad/Bl)':<24}"
    print(header)
    print("-" * len(header))

    seen_v3_0 = benchmark_summary["1_Seen_Structure"]["V3_0_DDQN"]["ir_mean"]
    seen_v3_1 = benchmark_summary["1_Seen_Structure"]["V3_1_DDQN"]["ir_mean"]
    seen_v4_0 = benchmark_summary["1_Seen_Structure"]["V4_0_Hybrid"]["ir_mean"]
    seen_v4_1_lstm = benchmark_summary["1_Seen_Structure"]["V4_1_LSTM_DDQN"]["ir_mean"]
    seen_v4_1_hyb = benchmark_summary["1_Seen_Structure"]["V4_1_LSTM_Hybrid"]["ir_mean"]

    ood_retention_dict: dict[str, Any] = {}

    for sc_name, sc_data in benchmark_summary.items():
        ca_ir = sc_data["Context_Aware"]["ir_mean"] * 100.0
        v30_ir = sc_data["V3_0_DDQN"]["ir_mean"] * 100.0
        v31_ir = sc_data["V3_1_DDQN"]["ir_mean"] * 100.0
        v40_ir = sc_data["V4_0_Hybrid"]["ir_mean"] * 100.0
        lstm_ir = sc_data["V4_1_LSTM_DDQN"]["ir_mean"] * 100.0
        v41_ir = sc_data["V4_1_LSTM_Hybrid"]["ir_mean"] * 100.0

        exp = sc_data["V4_1_LSTM_Hybrid"]["pct_ddqn_exploit"]
        ad = sc_data["V4_1_LSTM_Hybrid"]["pct_ca_adapt"]
        bl = sc_data["V4_1_LSTM_Hybrid"]["pct_blended"]

        modes_str = f"{exp:.0f}% / {ad:.0f}% / {bl:.0f}%"

        print(
            f"{sc_name:<22} | {ca_ir:5.1f}%  | {v30_ir:6.1f}%  | {v31_ir:6.1f}%  | {v40_ir:6.1f}%  | {lstm_ir:7.1f}%  | {v41_ir:6.1f}%  | {modes_str:<24}"
        )

        if sc_name != "1_Seen_Structure":
            ood_retention_dict[sc_name] = {
                "v3_0_retention": (v30_ir / max(seen_v3_0 * 100.0, 1e-4)) * 100.0,
                "v3_1_retention": (v31_ir / max(seen_v3_1 * 100.0, 1e-4)) * 100.0,
                "v4_0_retention": (v40_ir / max(seen_v4_0 * 100.0, 1e-4)) * 100.0,
                "v4_1_lstm_retention": (lstm_ir / max(seen_v4_1_lstm * 100.0, 1e-4)) * 100.0,
                "v4_1_hyb_retention": (v41_ir / max(seen_v4_1_hyb * 100.0, 1e-4)) * 100.0,
            }

    print("=" * 105)

    results_payload = {
        "benchmark_summary": benchmark_summary,
        "ood_retention": ood_retention_dict,
        "metadata": {
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "steps": 300,
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - t_start,
        },
    }

    out_path = Path("scratch/validation/v4_1_benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n✅ Results saved to {out_path} (elapsed: {time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
