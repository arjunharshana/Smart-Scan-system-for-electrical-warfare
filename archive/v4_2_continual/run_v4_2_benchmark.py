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
from rf_environment.scheduler.context_aware import ContextAwareScheduler
from rf_environment.scheduler.hybrid.arbitrator import ArbitrationMode
from rf_environment.scheduler.hybrid.continual.config import V42ContinualConfig
from rf_environment.scheduler.hybrid.continual.regime_detector import RegimeState
from rf_environment.scheduler.hybrid.hybrid_v41 import LSTMHybridScheduler
from rf_environment.scheduler.hybrid.hybrid_v42 import V42ContinualAdaptiveHybrid
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


def make_phase_a_scenario(total_time_steps: int = 300, seed: int = 42) -> dict[str, Any]:
    return {
        "simulation": {"total_time_steps": total_time_steps, "time_step_ms": 10, "seed": seed},
        "spectrum": SPECTRUM_CFG,
        "receiver": RECEIVER_CFG,
        "detector": DETECTOR_CFG,
        "scheduler": {"type": "sequential"},
        "emitters": [
            {
                "id": "EM_A", "type": "radar",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]],
                    "mode": "sequential",
                    "dwell_steps": 3,
                },
                "time_behavior": {"type": "continuous"},
                "power_dbm": -18, "bandwidth_hz": 2_000_000,
            }
        ],
    }


def make_synthetic_regime_shift_scenario(seed: int = 42) -> dict[str, Any]:
    """Creates a continuous 900-step 3-phase regime-shift scenario:
    Phase A: [0, 300) -> [5, 15, 25], dwell=3
    Phase B: [300, 600) -> [10, 20, 27], dwell=5
    Phase C: [600, 900) -> [5, 15, 25], dwell=3
    """
    return {
        "simulation": {"total_time_steps": 900, "time_step_ms": 10, "seed": seed},
        "spectrum": SPECTRUM_CFG,
        "receiver": RECEIVER_CFG,
        "detector": DETECTOR_CFG,
        "scheduler": {"type": "sequential"},
        "emitters": [
            # Phase A: [5, 15, 25], dwell=3, active t in [0, 300)
            {
                "id": "EM_A", "type": "radar",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]],
                    "mode": "sequential",
                    "dwell_steps": 3,
                },
                "time_behavior": {"type": "burst", "burst_duration": 300, "interval": 100000, "start_time": 0},
                "power_dbm": -18, "bandwidth_hz": 2_000_000,
            },
            # Phase B: [10, 20, 27], dwell=5, active t in [300, 600)
            {
                "id": "EM_B", "type": "radar",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [bin_to_center_hz(b) for b in [10, 20, 27]],
                    "mode": "sequential",
                    "dwell_steps": 5,
                },
                "time_behavior": {"type": "burst", "burst_duration": 300, "interval": 100000, "start_time": 300},
                "power_dbm": -18, "bandwidth_hz": 2_000_000,
            },
            # Phase C: [5, 15, 25], dwell=3, active t in [600, 900)
            {
                "id": "EM_C", "type": "radar",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [bin_to_center_hz(b) for b in [5, 15, 25]],
                    "mode": "sequential",
                    "dwell_steps": 3,
                },
                "time_behavior": {"type": "burst", "burst_duration": 300, "interval": 100000, "start_time": 600},
                "power_dbm": -18, "bandwidth_hz": 2_000_000,
            },
        ],
    }


def evaluate_900_step_run(
    scenario_cfg: dict[str, Any],
    scheduler: Any,
    seed: int,
) -> dict[str, Any]:
    sc = copy.deepcopy(scenario_cfg)
    sc["simulation"]["seed"] = seed
    env = build_environment(sc)

    # Initialize scheduler
    scheduler.eval()
    if hasattr(scheduler, "epsilon"):
        scheduler.epsilon = 0.0

    num_bins = len(env.bands_hz)

    class SafeWrapper:
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

        def get_diagnostics(self) -> dict[str, Any]:
            if hasattr(self._inner, "get_diagnostics"):
                return self._inner.get_diagnostics()
            return {}

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    wrapped = SafeWrapper(scheduler, num_bins)
    env.scheduler = wrapped
    wrapped.reset()

    step_results = env.run(steps=900)

    # Compute step-by-step metrics
    hits = [bool(r.observation.last_detection) for r in step_results]

    # Opportunities per step
    # We inspect ground truth opportunity records from environment metrics
    opp_records = env.metrics.opportunity_tracker.get_all_opportunities()
    opps_by_step = [0] * 900
    for opp in opp_records:
        for t in range(opp.start_step, min(opp.end_step + 1, 900)):
            opps_by_step[t] = 1

    def compute_ir(t_start: int, t_end: int) -> float:
        sub_hits = sum(hits[t_start:t_end])
        sub_opps = sum(opps_by_step[t_start:t_end])
        return float(sub_hits) / float(max(sub_opps, 1))

    # Phase Metrics
    ir_phase_a = compute_ir(0, 300)
    ir_phase_b_imm = compute_ir(300, 350)
    ir_phase_b_post = compute_ir(450, 600)
    ir_phase_b_all = compute_ir(300, 600)
    ir_phase_c = compute_ir(600, 900)
    ir_overall = compute_ir(0, 900)

    # Recovery Steps Metric:
    # Steps from t=300 until rolling 20-step window IR reaches 80% of Phase B post-adaptation IR
    recovery_target = max(ir_phase_b_post * 0.80, 0.10)
    recovery_step: int | None = None
    w_size = 20
    for t in range(300 + w_size, 600):
        w_ir = compute_ir(t - w_size, t)
        if w_ir >= recovery_target:
            recovery_step = t - 300
            break
    if recovery_step is None:
        recovery_step = 300  # failed to recover within Phase B

    # Catastrophic Forgetting Retention: Phase C IR / Phase A IR
    retention_c = (ir_phase_c / max(ir_phase_a, 1e-4)) * 100.0

    diag = wrapped.get_diagnostics()

    return {
        "ir_overall": ir_overall,
        "ir_phase_a": ir_phase_a,
        "ir_phase_b_imm": ir_phase_b_imm,
        "ir_phase_b_post": ir_phase_b_post,
        "ir_phase_b_all": ir_phase_b_all,
        "ir_phase_c": ir_phase_c,
        "retention_phase_c": retention_c,
        "adaptation_recovery_steps": recovery_step,
        "diagnostics": diag,
    }


def main() -> None:
    t_start = time.time()
    print("=" * 80)
    print("🚀 SIH26055 — V4.2 CONTINUAL ADAPTIVE HYBRID BENCHMARK")
    print("=" * 80)

    # 1. Pre-train baseline models on Phase A
    print("\n--- STEP 1: Pre-training Recurrent Models on Phase A ([5, 15, 25], dwell=3) ---")
    sc_phase_a = make_phase_a_scenario()
    lstm_models = []
    historical_buffers = []

    for t_seed in TRAIN_SEEDS:
        env_train = build_environment(copy.deepcopy(sc_phase_a))
        enc = TemporalObservationEncoder.create_encoder_b(num_bins=len(env_train.bands_hz))
        m_lstm = LSTMDDQNScheduler(
            bands_hz=env_train.bands_hz,
            encoder=enc,
            hidden_dim=64,
            dense_dim=64,
            sequence_length=10,
            learning_rate=0.001,
            seed=t_seed,
        )
        m_lstm.train()
        env_train.scheduler = m_lstm

        # 5 training episodes of 300 steps
        for ep in range(5):
            ep_seed = t_seed + ep * 1000
            s_eps = m_lstm.epsilon
            env_train.reset(seed=ep_seed)
            m_lstm.epsilon = s_eps
            env_train.run(steps=300)

        lstm_models.append(m_lstm)
        historical_buffers.append(copy.deepcopy(m_lstm.replay_buffer))

    print(f"   Trained {len(lstm_models)} recurrent models on Phase A.")

    # 2. Run 900-Step Continual Benchmark Across 10 Evaluation Seeds
    print("\n--- STEP 2: Running 900-Step 3-Phase Continual Benchmark ---")
    benchmark_results: dict[str, Any] = {
        "Context_Aware": [],
        "V4_1_LSTM_Hybrid_Frozen": [],
        "V4_2_Continual_Hybrid": [],
        "Standalone_LSTM_DDQN": [],
    }

    sc_shift = make_synthetic_regime_shift_scenario()

    for e_idx, e_s in enumerate(EVAL_SEEDS):
        print(f"\n[Eval Seed {e_s} ({e_idx + 1}/{len(EVAL_SEEDS)})]")

        # 1. Context-Aware
        env_ca = build_environment(copy.deepcopy(sc_shift))
        ca_sched = ContextAwareScheduler(bands_hz=env_ca.bands_hz, seed=e_s)
        res_ca = evaluate_900_step_run(sc_shift, ca_sched, seed=e_s)
        benchmark_results["Context_Aware"].append(res_ca)
        print(f"   Context-Aware:     Overall={res_ca['ir_overall']*100:.1f}%, Phase A={res_ca['ir_phase_a']*100:.1f}%, Phase B={res_ca['ir_phase_b_all']*100:.1f}%, Phase C={res_ca['ir_phase_c']*100:.1f}%")

        # 2. Standalone LSTM-DDQN (Ensemble average)
        s_res_lstm = []
        for m in lstm_models:
            m_copy = copy.deepcopy(m)
            m_copy.eval()
            m_copy.epsilon = 0.0
            r = evaluate_900_step_run(sc_shift, m_copy, seed=e_s)
            s_res_lstm.append(r)
        avg_lstm = {
            "ir_overall": float(np.mean([r["ir_overall"] for r in s_res_lstm])),
            "ir_phase_a": float(np.mean([r["ir_phase_a"] for r in s_res_lstm])),
            "ir_phase_b_imm": float(np.mean([r["ir_phase_b_imm"] for r in s_res_lstm])),
            "ir_phase_b_post": float(np.mean([r["ir_phase_b_post"] for r in s_res_lstm])),
            "ir_phase_b_all": float(np.mean([r["ir_phase_b_all"] for r in s_res_lstm])),
            "ir_phase_c": float(np.mean([r["ir_phase_c"] for r in s_res_lstm])),
            "retention_phase_c": float(np.mean([r["retention_phase_c"] for r in s_res_lstm])),
            "adaptation_recovery_steps": float(np.mean([r["adaptation_recovery_steps"] for r in s_res_lstm])),
            "diagnostics": s_res_lstm[0]["diagnostics"],
        }
        benchmark_results["Standalone_LSTM_DDQN"].append(avg_lstm)
        print(f"   Standalone LSTM:   Overall={avg_lstm['ir_overall']*100:.1f}%, Phase A={avg_lstm['ir_phase_a']*100:.1f}%, Phase B={avg_lstm['ir_phase_b_all']*100:.1f}%, Phase C={avg_lstm['ir_phase_c']*100:.1f}%")

        # 3. V4.1 LSTM-Hybrid Frozen (Ensemble average)
        s_res_v41 = []
        for m in lstm_models:
            env_tmp = build_environment(copy.deepcopy(sc_shift))
            m_copy = copy.deepcopy(m)
            m_copy.bands_hz = env_tmp.bands_hz
            m_copy.eval()
            m_copy.epsilon = 0.0

            h41 = LSTMHybridScheduler(
                bands_hz=env_tmp.bands_hz,
                lstm_ddqn=m_copy,
                seed=derive_seed(e_s, "v41"),
            )
            h41.eval()
            r = evaluate_900_step_run(sc_shift, h41, seed=e_s)
            s_res_v41.append(r)
        avg_v41 = {
            "ir_overall": float(np.mean([r["ir_overall"] for r in s_res_v41])),
            "ir_phase_a": float(np.mean([r["ir_phase_a"] for r in s_res_v41])),
            "ir_phase_b_imm": float(np.mean([r["ir_phase_b_imm"] for r in s_res_v41])),
            "ir_phase_b_post": float(np.mean([r["ir_phase_b_post"] for r in s_res_v41])),
            "ir_phase_b_all": float(np.mean([r["ir_phase_b_all"] for r in s_res_v41])),
            "ir_phase_c": float(np.mean([r["ir_phase_c"] for r in s_res_v41])),
            "retention_phase_c": float(np.mean([r["retention_phase_c"] for r in s_res_v41])),
            "adaptation_recovery_steps": float(np.mean([r["adaptation_recovery_steps"] for r in s_res_v41])),
            "diagnostics": s_res_v41[0]["diagnostics"],
        }
        benchmark_results["V4_1_LSTM_Hybrid_Frozen"].append(avg_v41)
        print(f"   V4.1 LSTM-Hybrid:  Overall={avg_v41['ir_overall']*100:.1f}%, Phase A={avg_v41['ir_phase_a']*100:.1f}%, Phase B={avg_v41['ir_phase_b_all']*100:.1f}%, Phase C={avg_v41['ir_phase_c']*100:.1f}%")

        # 4. V4.2 Continual Adaptive Hybrid (Ensemble average)
        s_res_v42 = []
        for m, h_buf in zip(lstm_models, historical_buffers):
            env_tmp = build_environment(copy.deepcopy(sc_shift))
            m_copy = copy.deepcopy(m)
            m_copy.bands_hz = env_tmp.bands_hz
            m_copy.eval()
            m_copy.epsilon = 0.0

            cfg_v42 = V42ContinualConfig(
                adaptation_updates=15,
                adaptation_learning_rate=0.0005,
                adaptation_cooldown_steps=40,
                min_replay_sequences=24,
            )

            h42 = V42ContinualAdaptiveHybrid(
                bands_hz=env_tmp.bands_hz,
                lstm_ddqn=m_copy,
                config=cfg_v42,
                seed=derive_seed(e_s, "v42"),
            )
            # Pre-seed protected historical reservoir with Phase A experiences
            h42.populate_historical_experience(h_buf)
            h42.eval()

            r = evaluate_900_step_run(sc_shift, h42, seed=e_s)
            s_res_v42.append(r)

        avg_v42 = {
            "ir_overall": float(np.mean([r["ir_overall"] for r in s_res_v42])),
            "ir_phase_a": float(np.mean([r["ir_phase_a"] for r in s_res_v42])),
            "ir_phase_b_imm": float(np.mean([r["ir_phase_b_imm"] for r in s_res_v42])),
            "ir_phase_b_post": float(np.mean([r["ir_phase_b_post"] for r in s_res_v42])),
            "ir_phase_b_all": float(np.mean([r["ir_phase_b_all"] for r in s_res_v42])),
            "ir_phase_c": float(np.mean([r["ir_phase_c"] for r in s_res_v42])),
            "retention_phase_c": float(np.mean([r["retention_phase_c"] for r in s_res_v42])),
            "adaptation_recovery_steps": float(np.mean([r["adaptation_recovery_steps"] for r in s_res_v42])),
            "diagnostics": s_res_v42[0]["diagnostics"],
        }
        benchmark_results["V4_2_Continual_Hybrid"].append(avg_v42)
        print(f"   V4.2 Continual:    Overall={avg_v42['ir_overall']*100:.1f}%, Phase A={avg_v42['ir_phase_a']*100:.1f}%, Phase B={avg_v42['ir_phase_b_all']*100:.1f}%, Phase C={avg_v42['ir_phase_c']*100:.1f}%")

    # 3. Aggregate Summary Across All Eval Seeds
    summary: dict[str, Any] = {}
    for name, runs in benchmark_results.items():
        summary[name] = {
            "ir_overall_mean": float(np.mean([r["ir_overall"] for r in runs])),
            "ir_overall_std": float(np.std([r["ir_overall"] for r in runs])),
            "ir_phase_a_mean": float(np.mean([r["ir_phase_a"] for r in runs])),
            "ir_phase_a_std": float(np.std([r["ir_phase_a"] for r in runs])),
            "ir_phase_b_imm_mean": float(np.mean([r["ir_phase_b_imm"] for r in runs])),
            "ir_phase_b_post_mean": float(np.mean([r["ir_phase_b_post"] for r in runs])),
            "ir_phase_b_all_mean": float(np.mean([r["ir_phase_b_all"] for r in runs])),
            "ir_phase_c_mean": float(np.mean([r["ir_phase_c"] for r in runs])),
            "retention_phase_c_mean": float(np.mean([r["retention_phase_c"] for r in runs])),
            "recovery_steps_mean": float(np.mean([r["adaptation_recovery_steps"] for r in runs])),
            "telemetry": runs[0]["diagnostics"],
        }

    print("\n" + "=" * 105)
    print("📊 V4.2 CONTINUAL ADAPTIVE HYBRID BENCHMARK SUMMARY")
    print("=" * 105)
    header = f"{'Scheduler':<25} | {'Overall IR':<11} | {'Phase A (Pre)':<13} | {'Phase B (Imm)':<13} | {'Phase B (Post)':<14} | {'Phase C (Ret)':<13} | {'Recov Steps':<11}"
    print(header)
    print("-" * len(header))

    for name, s in summary.items():
        ov = f"{s['ir_overall_mean']*100:.1f}%"
        pa = f"{s['ir_phase_a_mean']*100:.1f}%"
        pb_imm = f"{s['ir_phase_b_imm_mean']*100:.1f}%"
        pb_post = f"{s['ir_phase_b_post_mean']*100:.1f}%"
        pc = f"{s['ir_phase_c_mean']*100:.1f}% ({s['retention_phase_c_mean']:.1f}%)"
        rec = f"{s['recovery_steps_mean']:.1f}"
        print(f"{name:<25} | {ov:<11} | {pa:<13} | {pb_imm:<13} | {pb_post:<14} | {pc:<13} | {rec:<11}")

    print("=" * 105)

    # Save to JSON
    out_payload = {
        "summary": summary,
        "per_seed_runs": benchmark_results,
        "metadata": {
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "phases": {
                "Phase_A": "0..300 steps, [5, 15, 25], dwell=3",
                "Phase_B": "300..600 steps, [10, 20, 27], dwell=5",
                "Phase_C": "600..900 steps, [5, 15, 25], dwell=3",
            },
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - t_start,
        },
    }

    out_path = Path("scratch/validation/v4_2_benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\n✅ Results saved to {out_path} (elapsed: {time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
