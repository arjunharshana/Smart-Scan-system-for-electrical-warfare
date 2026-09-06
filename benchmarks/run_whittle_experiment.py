"""benchmarks/run_whittle_experiment.py
Controlled Research Experiment: Standalone Whittle-Style Heuristic Index Scheduler Evaluation.

Objectives:
1. Validation Tuning Protocol:
   - Perform grid search tuning strictly on held-out validation scenarios VAL_1..4 (seeds [10, 20, 30]).
   - Tune uncertainty weight lambda in {0.10, 0.20, 0.35}, recency/transition weight mu in {0.15, 0.35, 0.55},
     and belief decay gamma in {0.90, 0.95, 0.99}.
   - Select best configuration using composite score:
     Score = mean(VAL_IR) - 0.5 * std(VAL_IR) - (0.20 if min(VAL_IR) < 0.10 else 0.0).
2. Ablation Study:
   - Evaluate W0 (Belief only), W1 (Belief + Recency), W2 (Belief + Recency + Uncertainty), and
     W3 (Full Model: Belief + Recency + Uncertainty + Dwell + Periodic).
3. Untouched Canonical Test Suite:
   - Evaluate promoted Whittle-style configuration across all 8 canonical test scenarios over
     10 seeds [10, 15, 20, 25, 30, 35, 40, 45, 50, 55] (300 steps).
   - Directly compare with Context-Aware (CA) baseline, V4.0 Hybrid, and V4.1 LSTM-Hybrid (H=64).
4. Latency & Computational Benchmark:
   - Measure per-decision latency (microseconds) and decisions/second for Whittle vs LSTM-DDQN vs CA.
5. Deliverables:
   - Save results to whittle_style_results.json and models/ablation/whittle_style_results.json.
   - Generate comprehensive markdown report whittle_style_report.md.
"""

from __future__ import annotations

import copy
import datetime
import itertools
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.scenarios_v4_1 import (
    get_canonical_test_scenarios,
    get_validation_scenarios,
    verify_dataset_isolation,
)
from rf_environment.domain.state import SchedulerObservation
from rf_environment.environment.builder import build_environment
from rf_environment.scheduler.context_aware import ContextAwareScheduler
from rf_environment.scheduler.whittle.config import WhittleConfig
from rf_environment.scheduler.whittle.scheduler import WhittleScheduler


def evaluate_whittle_on_scenario(
    scenario_dict: dict[str, Any],
    config: WhittleConfig | dict[str, Any],
    seeds: list[int],
    steps: int = 300,
) -> dict[str, Any]:
    """Runs WhittleScheduler on a scenario across multiple seeds and aggregates metrics."""
    irs: list[float] = []
    det_rates: list[float] = []
    scan_effs: list[float] = []

    for s in seeds:
        sc_copy = copy.deepcopy(scenario_dict)
        sc_copy["simulation"]["seed"] = s
        sc_copy["simulation"]["total_time_steps"] = steps

        env = build_environment(sc_copy)
        sched = WhittleScheduler(env.bands_hz, config=config, seed=s)
        env.scheduler = sched
        sched.reset()

        step_results = env.run(steps=steps)
        snap = env.metrics.snapshot(env.clock.time_step).to_dict()

        ir = float(snap.get("interception_ratio", 0.0))
        det_rate = float(snap.get("probability_of_detection", 0.0))
        hits = sum(1 for r in step_results if r.observation.last_detection)
        eff = float(hits) / float(max(len(step_results), 1))

        irs.append(ir)
        det_rates.append(det_rate)
        scan_effs.append(eff)

    return {
        "ir_mean": float(np.mean(irs)),
        "ir_std": float(np.std(irs)),
        "det_rate_mean": float(np.mean(det_rates)),
        "scan_eff_mean": float(np.mean(scan_effs)),
        "per_seed_ir": irs,
    }


def evaluate_scheduler_on_scenario(
    scenario_dict: dict[str, Any],
    scheduler_name: str,
    seeds: list[int],
    steps: int = 300,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluates an arbitrary registered scheduler across seeds."""
    irs: list[float] = []
    det_rates: list[float] = []
    scan_effs: list[float] = []

    for s in seeds:
        sc_copy = copy.deepcopy(scenario_dict)
        sc_copy["simulation"]["seed"] = s
        sc_copy["simulation"]["total_time_steps"] = steps

        env = build_environment(sc_copy, scheduler_name=scheduler_name, **kwargs)
        if hasattr(env.scheduler, "eval"):
            env.scheduler.eval()
        if hasattr(env.scheduler, "epsilon"):
            env.scheduler.epsilon = 0.0
        if hasattr(env.scheduler, "reset"):
            env.scheduler.reset()

        step_results = env.run(steps=steps)
        snap = env.metrics.snapshot(env.clock.time_step).to_dict()

        ir = float(snap.get("interception_ratio", 0.0))
        det_rate = float(snap.get("probability_of_detection", 0.0))
        hits = sum(1 for r in step_results if r.observation.last_detection)
        eff = float(hits) / float(max(len(step_results), 1))

        irs.append(ir)
        det_rates.append(det_rate)
        scan_effs.append(eff)

    return {
        "ir_mean": float(np.mean(irs)),
        "ir_std": float(np.std(irs)),
        "det_rate_mean": float(np.mean(det_rates)),
        "scan_eff_mean": float(np.mean(scan_effs)),
        "per_seed_ir": irs,
    }


def run_validation_battery(
    config: WhittleConfig | dict[str, Any],
    val_scenarios: dict[str, dict[str, Any]],
    val_seeds: list[int],
    steps: int = 300,
) -> dict[str, Any]:
    """Evaluates a Whittle configuration across all held-out validation scenarios."""
    sc_metrics: dict[str, Any] = {}
    all_irs: list[float] = []
    all_dets: list[float] = []

    for sc_name, sc_cfg in val_scenarios.items():
        res = evaluate_whittle_on_scenario(sc_cfg, config, val_seeds, steps=steps)
        sc_metrics[sc_name] = res
        all_irs.append(res["ir_mean"])
        all_dets.append(res["det_rate_mean"])

    ir_mean = float(np.mean(all_irs))
    ir_std = float(np.std(all_irs))
    ir_min = float(np.min(all_irs))
    composite_score = ir_mean - 0.5 * ir_std - (0.20 if ir_min < 0.10 else 0.0)

    return {
        "val_ir_mean": ir_mean,
        "val_ir_std": ir_std,
        "val_ir_min": ir_min,
        "val_det_rate": float(np.mean(all_dets)),
        "composite_score": float(composite_score),
        "scenario_metrics": sc_metrics,
    }


def run_latency_benchmark(num_steps: int = 5000) -> dict[str, Any]:
    """Benchmarks per-decision latency and throughput for Whittle vs CA vs LSTM-DDQN."""
    num_bins = 30
    bands = [100e6 + 20e6 * i for i in range(num_bins)]

    # Create schedulers
    whittle = WhittleScheduler(bands, config=WhittleConfig(ablation_mode="W3"))
    ca = ContextAwareScheduler(bands, seed=42)

    # Dummy observation
    def _make_obs(step: int, hit: bool, b: int) -> SchedulerObservation:
        return SchedulerObservation(
            timestamp=float(step),
            current_frequency_bin=b,
            last_detection=hit,
            last_detection_bin=b if hit else None,
            last_detection_strength=-65.0 if hit else None,
            recent_detection_history=tuple([hit] * 5),
            recent_frequency_history=tuple([b] * 5),
            scan_count_by_bin=tuple([1] * num_bins),
            time_since_scan_by_bin=tuple([1.0] * num_bins),
            time_since_last_detection=0.0 if hit else 1.0,
        )

    # Warmup
    for i in range(100):
        obs = _make_obs(i, i % 3 == 0, (i * 3) % num_bins)
        whittle.select_bin(obs)
        ca.select_bin(obs)

    # Benchmark Whittle
    whittle.reset()
    latencies_whittle: list[float] = []
    t0 = time.perf_counter()
    for i in range(num_steps):
        obs = _make_obs(i, i % 3 == 0, (i * 3) % num_bins)
        t_start = time.perf_counter()
        whittle.select_bin(obs)
        latencies_whittle.append(time.perf_counter() - t_start)
    whittle_total_s = time.perf_counter() - t0

    # Benchmark CA
    ca.reset()
    latencies_ca: list[float] = []
    t0 = time.perf_counter()
    for i in range(num_steps):
        obs = _make_obs(i, i % 3 == 0, (i * 3) % num_bins)
        t_start = time.perf_counter()
        ca.select_bin(obs)
        latencies_ca.append(time.perf_counter() - t_start)
    ca_total_s = time.perf_counter() - t0

    # Latency for LSTM if production checkpoint available
    ckpt_path = PROJECT_ROOT / "models/v4_1/production_checkpoint.npz"
    lstm_metrics: dict[str, Any] = {}
    if ckpt_path.exists():
        from rf_environment.scheduler.hybrid.hybrid_v41 import LSTMHybridScheduler
        hybrid = LSTMHybridScheduler(bands, checkpoint_path=str(ckpt_path), require_checkpoint=True)
        hybrid.eval()
        hybrid.reset()
        for i in range(50):
            hybrid.select_action(_make_obs(i, False, 0))
        latencies_hybrid: list[float] = []
        t0 = time.perf_counter()
        for i in range(min(num_steps, 2000)):
            obs = _make_obs(i, i % 3 == 0, (i * 3) % num_bins)
            t_start = time.perf_counter()
            hybrid.select_action(obs)
            latencies_hybrid.append(time.perf_counter() - t_start)
        hybrid_total_s = time.perf_counter() - t0
        lstm_metrics = {
            "mean_latency_us": float(np.mean(latencies_hybrid) * 1e6),
            "p95_latency_us": float(np.percentile(latencies_hybrid, 95) * 1e6),
            "p99_latency_us": float(np.percentile(latencies_hybrid, 99) * 1e6),
            "decisions_per_sec": float(len(latencies_hybrid) / hybrid_total_s),
        }

    return {
        "num_steps": num_steps,
        "whittle": {
            "mean_latency_us": float(np.mean(latencies_whittle) * 1e6),
            "p95_latency_us": float(np.percentile(latencies_whittle, 95) * 1e6),
            "p99_latency_us": float(np.percentile(latencies_whittle, 99) * 1e6),
            "decisions_per_sec": float(num_steps / whittle_total_s),
        },
        "context_aware": {
            "mean_latency_us": float(np.mean(latencies_ca) * 1e6),
            "p95_latency_us": float(np.percentile(latencies_ca, 95) * 1e6),
            "p99_latency_us": float(np.percentile(latencies_ca, 99) * 1e6),
            "decisions_per_sec": float(num_steps / ca_total_s),
        },
        "v41_lstm_hybrid": lstm_metrics,
    }


def run_experiment() -> dict[str, Any]:
    t_start = time.time()
    print("=" * 80)
    print("🚀 STANDALONE WHITTLE-STYLE HEURISTIC INDEX SCHEDULER EXPERIMENT")
    print("=" * 80)

    # 0. Dataset Isolation Verification
    assert verify_dataset_isolation(), "Dataset isolation check failed!"
    print("✅ Dataset Isolation Verified (Zero Leakage between Train, Val, and Canonical Test)")

    val_scenarios = get_validation_scenarios()
    val_seeds = [10, 20, 30]
    test_scenarios = get_canonical_test_scenarios()
    eval_seeds = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

    # =========================================================================
    # PHASE 1: VALIDATION TUNING PROTOCOL (Held-out VAL_1..4)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🔍 PHASE 1: VALIDATION TUNING (GRID SEARCH ON HELD-OUT VAL_1..4)")
    print("=" * 80)

    lambda_vals = [0.10, 0.20, 0.35]
    mu_vals = [0.15, 0.35, 0.55]
    gamma_vals = [0.90, 0.95, 0.99]

    tuning_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_score = -float("inf")

    total_configs = len(lambda_vals) * len(mu_vals) * len(gamma_vals)
    cfg_idx = 0

    for lam, mu, gam in itertools.product(lambda_vals, mu_vals, gamma_vals):
        cfg_idx += 1
        cfg = WhittleConfig(
            w_uncertainty=lam,
            w_recency=mu,
            belief_decay=gam,
            w_dwell=0.15,
            w_periodic=0.15,
            stale_threshold=10,
            ablation_mode="W3",
        )
        val_res = run_validation_battery(cfg, val_scenarios, val_seeds, steps=300)
        rec = {
            "config_id": f"CFG_{cfg_idx:02d}",
            "weight_uncertainty": lam,
            "weight_recency": mu,
            "belief_decay": gam,
            "val_ir_mean": val_res["val_ir_mean"],
            "val_ir_std": val_res["val_ir_std"],
            "val_ir_min": val_res["val_ir_min"],
            "val_det_rate": val_res["val_det_rate"],
            "composite_score": val_res["composite_score"],
        }
        tuning_records.append(rec)

        if val_res["composite_score"] > best_score:
            best_score = val_res["composite_score"]
            best_record = rec

        print(
            f"[{cfg_idx:2d}/{total_configs:2d}] λ={lam:4.2f}, μ={mu:4.2f}, γ={gam:4.2f} | "
            f"Val IR: {val_res['val_ir_mean']*100:5.2f}% (±{val_res['val_ir_std']*100:4.2f}%, min={val_res['val_ir_min']*100:4.1f}%) | "
            f"Score: {val_res['composite_score']:.4f}"
        )

    assert best_record is not None
    print("\n" + "-" * 80)
    print(f"🏆 BEST VALIDATION CONFIGURATION SELECTED: {best_record['config_id']}")
    print(f"   λ (Uncertainty): {best_record['weight_uncertainty']}")
    print(f"   μ (Recency):     {best_record['weight_recency']}")
    print(f"   γ (Decay):       {best_record['belief_decay']}")
    print(f"   Validation IR:   {best_record['val_ir_mean']*100:.2f}% (±{best_record['val_ir_std']*100:.2f}%)")
    print(f"   Composite Score: {best_record['composite_score']:.4f}")
    print("-" * 80)

    promoted_config = WhittleConfig(
        w_uncertainty=best_record["weight_uncertainty"],
        w_recency=best_record["weight_recency"],
        belief_decay=best_record["belief_decay"],
        w_dwell=0.15,
        w_periodic=0.15,
        stale_threshold=10,
        ablation_mode="W3",
    )

    # =========================================================================
    # PHASE 2: ABLATION STUDY (W0, W1, W2, W3)
    # =========================================================================
    print("\n" + "=" * 80)
    print("🔬 PHASE 2: ABLATION STUDY (W0, W1, W2, W3)")
    print("=" * 80)

    ablations = {
        "W0": WhittleConfig(
            ablation_mode="W0",
            belief_decay=best_record["belief_decay"],
        ),
        "W1": WhittleConfig(
            ablation_mode="W1",
            w_recency=best_record["weight_recency"],
            belief_decay=best_record["belief_decay"],
        ),
        "W2": WhittleConfig(
            ablation_mode="W2",
            w_recency=best_record["weight_recency"],
            w_uncertainty=best_record["weight_uncertainty"],
            belief_decay=best_record["belief_decay"],
            stale_threshold=10,
        ),
        "W3": promoted_config,
    }

    ablation_results: dict[str, Any] = {}
    for mode_name, ab_cfg in ablations.items():
        print(f"\nEvaluating Ablation Mode {mode_name}...")
        # 1. Val battery
        val_res = run_validation_battery(ab_cfg, val_scenarios, val_seeds, steps=300)
        # 2. Canonical test suite
        test_sc_res: dict[str, Any] = {}
        all_test_irs: list[float] = []
        all_test_dets: list[float] = []
        for sc_name, sc_cfg in test_scenarios.items():
            sc_res = evaluate_whittle_on_scenario(sc_cfg, ab_cfg, eval_seeds, steps=300)
            test_sc_res[sc_name] = sc_res
            all_test_irs.append(sc_res["ir_mean"])
            all_test_dets.append(sc_res["det_rate_mean"])

        ab_record = {
            "mode": mode_name,
            "description": {
                "W0": "Belief Only (pi_i)",
                "W1": "Belief + Recency (pi_i + mu * T_hat)",
                "W2": "Belief + Recency + Uncertainty (pi_i + mu * T_hat + lambda * u_i)",
                "W3": "Full Model (Belief + Recency + Uncertainty + Dwell + Periodic)",
            }[mode_name],
            "validation": {
                "val_ir_mean": val_res["val_ir_mean"],
                "val_ir_std": val_res["val_ir_std"],
                "val_det_rate": val_res["val_det_rate"],
            },
            "canonical_test": {
                "test_ir_mean": float(np.mean(all_test_irs)),
                "test_ir_std": float(np.std(all_test_irs)),
                "test_det_rate": float(np.mean(all_test_dets)),
                "per_scenario": test_sc_res,
            },
        }
        ablation_results[mode_name] = ab_record
        print(
            f"   Mode {mode_name}: Val IR = {val_res['val_ir_mean']*100:5.2f}% | "
            f"Canonical Test IR = {np.mean(all_test_irs)*100:5.2f}% (±{np.std(all_test_irs)*100:4.2f}%)"
        )

    # =========================================================================
    # PHASE 3: CANONICAL TEST SUITE HEAD-TO-HEAD COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 PHASE 3: CANONICAL TEST SUITE BENCHMARK (HEAD-TO-HEAD COMPARISON)")
    print("=" * 80)

    # Checkpoint path for V4.1
    ckpt_path = PROJECT_ROOT / "models/v4_1/production_checkpoint.npz"
    v4_benchmark_file = PROJECT_ROOT / "benchmarks/results/v4_benchmark_results.json"

    # Load historical V4.0 results if available
    v40_results_by_scenario: dict[str, float] = {}
    if v4_benchmark_file.exists():
        with open(v4_benchmark_file, "r", encoding="utf-8") as f:
            raw_v4 = json.load(f).get("benchmark_summary", {})
            for sc_k, sc_v in raw_v4.items():
                if "V4_0_Hybrid" in sc_v:
                    v40_results_by_scenario[sc_k] = float(sc_v["V4_0_Hybrid"].get("ir_mean", 0.0))

    scenario_comparison_table: dict[str, Any] = {}
    ca_test_irs: list[float] = []
    v41_test_irs: list[float] = []
    whittle_test_irs: list[float] = []

    print(
        f"{'Scenario':<22} | {'CA Baseline':<11} | {'V4.0 Hybrid':<11} | "
        f"{'V4.1 (H=64)':<11} | {'Whittle (W3)':<12} | {'Whittle vs V4.1':<15}"
    )
    print("-" * 90)

    for sc_name, sc_cfg in test_scenarios.items():
        # 1. Context-Aware Baseline
        ca_res = evaluate_scheduler_on_scenario(sc_cfg, "context_aware", eval_seeds, steps=300)
        ca_test_irs.append(ca_res["ir_mean"])

        # 2. V4.0 Hybrid
        v40_ir = v40_results_by_scenario.get(sc_name, float("nan"))

        # 3. V4.1 LSTM-Hybrid (H=64) with production checkpoint
        v41_res = evaluate_scheduler_on_scenario(
            sc_cfg,
            "hybrid_v41",
            eval_seeds,
            steps=300,
            checkpoint_path=str(ckpt_path),
            require_checkpoint=True,
        )
        v41_test_irs.append(v41_res["ir_mean"])

        # 4. Whittle-style (W3 Promoted)
        whittle_res = ablation_results["W3"]["canonical_test"]["per_scenario"][sc_name]
        whittle_test_irs.append(whittle_res["ir_mean"])

        diff_vs_v41 = (whittle_res["ir_mean"] - v41_res["ir_mean"]) * 100.0
        diff_str = f"{diff_vs_v41:+5.2f}%"

        v40_str = f"{v40_ir*100:5.2f}%" if not np.isnan(v40_ir) else "  N/A "

        print(
            f"{sc_name:<22} | {ca_res['ir_mean']*100:5.2f}%     | {v40_str:<11} | "
            f"{v41_res['ir_mean']*100:5.2f}%     | {whittle_res['ir_mean']*100:5.2f}%      | {diff_str:<15}"
        )

        scenario_comparison_table[sc_name] = {
            "ca_baseline": ca_res,
            "v4_0_hybrid_ir_mean": v40_ir,
            "v4_1_hybrid": v41_res,
            "whittle_w3": whittle_res,
            "delta_ir_pct_vs_ca": (whittle_res["ir_mean"] - ca_res["ir_mean"]) * 100.0,
            "delta_ir_pct_vs_v41": diff_vs_v41,
        }

    print("-" * 90)
    print(
        f"{'OVERALL MEAN':<22} | {np.mean(ca_test_irs)*100:5.2f}%     | "
        f"{np.nanmean(list(v40_results_by_scenario.values()))*100:5.2f}%     | "
        f"{np.mean(v41_test_irs)*100:5.2f}%     | {np.mean(whittle_test_irs)*100:5.2f}%      | "
        f"{(np.mean(whittle_test_irs) - np.mean(v41_test_irs))*100:+5.2f}%"
    )

    # =========================================================================
    # PHASE 4: LATENCY & COMPUTATIONAL BENCHMARK
    # =========================================================================
    print("\n" + "=" * 80)
    print("⚡ PHASE 4: LATENCY & COMPUTATIONAL BENCHMARK")
    print("=" * 80)

    latency_data = run_latency_benchmark(num_steps=3000)
    print(f"Whittle Decision Latency:  {latency_data['whittle']['mean_latency_us']:.2f} μs | Throughput: {latency_data['whittle']['decisions_per_sec']:.0f} decisions/sec")
    print(f"Context-Aware Latency:     {latency_data['context_aware']['mean_latency_us']:.2f} μs | Throughput: {latency_data['context_aware']['decisions_per_sec']:.0f} decisions/sec")
    if latency_data["v41_lstm_hybrid"]:
        print(f"V4.1 LSTM Hybrid Latency:  {latency_data['v41_lstm_hybrid']['mean_latency_us']:.2f} μs | Throughput: {latency_data['v41_lstm_hybrid']['decisions_per_sec']:.0f} decisions/sec")

    # =========================================================================
    # DELIVERABLE EXPORT
    # =========================================================================
    total_elapsed = time.time() - t_start

    final_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_elapsed_seconds": total_elapsed,
        "tuning_protocol": {
            "grid_lambda": lambda_vals,
            "grid_mu": mu_vals,
            "grid_gamma": gamma_vals,
            "val_seeds": val_seeds,
            "promoted_config": {
                "config_id": best_record["config_id"],
                "weight_uncertainty": best_record["weight_uncertainty"],
                "weight_recency": best_record["weight_recency"],
                "belief_decay": best_record["belief_decay"],
                "val_ir_mean": best_record["val_ir_mean"],
                "val_ir_std": best_record["val_ir_std"],
                "composite_score": best_record["composite_score"],
            },
            "all_tuning_records": tuning_records,
        },
        "ablation_results": ablation_results,
        "scenario_comparison": scenario_comparison_table,
        "summary": {
            "ca_test_ir_mean": float(np.mean(ca_test_irs)),
            "ca_test_ir_std": float(np.std(ca_test_irs)),
            "v40_test_ir_mean": float(np.nanmean(list(v40_results_by_scenario.values()))),
            "v41_test_ir_mean": float(np.mean(v41_test_irs)),
            "v41_test_ir_std": float(np.std(v41_test_irs)),
            "whittle_test_ir_mean": float(np.mean(whittle_test_irs)),
            "whittle_test_ir_std": float(np.std(whittle_test_irs)),
            "delta_whittle_vs_ca": float(np.mean(whittle_test_irs) - np.mean(ca_test_irs)),
            "delta_whittle_vs_v41": float(np.mean(whittle_test_irs) - np.mean(v41_test_irs)),
        },
        "latency_benchmark": latency_data,
    }

    # Save outputs
    out_root = PROJECT_ROOT / "whittle_style_results.json"
    out_ablation = PROJECT_ROOT / "models/ablation/whittle_style_results.json"
    out_ablation.parent.mkdir(parents=True, exist_ok=True)

    with open(out_root, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    with open(out_ablation, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\n✅ Results exported to:\n  - {out_root}\n  - {out_ablation}")
    print(f"✨ Standalone Whittle Evaluation Completed in {total_elapsed:.1f}s.")
    return final_payload


if __name__ == "__main__":
    run_experiment()
