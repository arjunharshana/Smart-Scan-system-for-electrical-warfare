"""benchmarks/run_capacity_ablation.py
Controlled Research Ablation: Recurrent Capacity Ablation (C64 vs C128 vs C256).

Objective:
Determine whether increasing recurrent working memory capacity (hidden size H in {64, 128, 256})
mitigates the performance degradation observed when training V4.1 LSTM-DDQN on 6 diverse regimes.

Configuration:
- Best replay configuration R2:
  * Stored hidden state = ON
  * Burn-in = 0
  * Regime-balanced replay = OFF
  * Sequence length = 10, Batch size = 32, lr = 0.001, target update = 100, eps decay = 0.9992
- Training distribution:
  * Full 6 diverse regimes (TRAIN_1..6)
  * 4 training seeds: [42, 123, 456, 789]
  * 30 episodes, 300 steps per episode
  * Validation every 5 episodes on held-out VAL_1..4
- Promotion rule:
  * Best validation objective score = val_ir_mean - 0.5 * val_ir_std - (0.20 if val_ir_min < 0.10 else 0.0)
- Untouched Canonical Test Suite:
  * Evaluated on all 8 canonical test scenarios across 10 evaluation seeds (300 steps).
- Hardware Constraint:
  * Mandatory GPU acceleration on CUDA (NVIDIA GeForce RTX 2050).
  * Lightweight pure-NumPy inference during validation and canonical testing.
"""

from __future__ import annotations

import copy
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.scenarios_v4_1 import (
    get_canonical_test_scenarios,
    get_train_scenarios,
    get_validation_scenarios,
    verify_dataset_isolation,
)
from benchmarks.train_v4_1 import (
    evaluate_canonical_test_suite,
    evaluate_hybrid_on_scenario,
    run_validation_battery,
    select_best_candidate,
)
from rf_environment.environment.builder import build_environment
from rf_environment.scheduler.rl.checkpoint_validator import verify_checkpoint_file_equivalence
from rf_environment.scheduler.rl.temporal_encoder import TemporalObservationEncoder
from rf_environment.scheduler.rl.torch_lstm_ddqn_scheduler import TorchLSTMDDQNScheduler


def get_hardware_audit() -> dict[str, Any]:
    """Inspects and verifies CUDA GPU availability and hardware specs."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CRITICAL: CUDA GPU is not available! GPU acceleration is a strict requirement for this ablation."
        )

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    device_props = torch.cuda.get_device_properties(0)
    vram_total_mb = int(device_props.total_memory / (1024 * 1024))

    # Driver & CUDA version query
    driver_version = "Unknown"
    cuda_version = torch.version.cuda or "Unknown"
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if smi_out:
            driver_version = smi_out.splitlines()[0].strip()
    except Exception:
        pass

    return {
        "gpu_available": True,
        "device_count": device_count,
        "device_name": device_name,
        "vram_total_mb": vram_total_mb,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "pytorch_version": torch.__version__,
        "cpu_count": os.cpu_count() or 1,
    }


def print_hardware_banner(hw: dict[str, Any]) -> None:
    print("=" * 80)
    print("🖥️  HARDWARE & GPU ACCELERATION AUDIT")
    print(f"   GPU Detected:        {hw['device_name']} ({hw['vram_total_mb']} MiB VRAM)")
    print(f"   Driver / CUDA:       Driver {hw['driver_version']} | CUDA {hw['cuda_version']}")
    print(f"   PyTorch Version:     {hw['pytorch_version']} (CUDA Active: True)")
    print(f"   CPU Threads:         {hw['cpu_count']}")
    print("=" * 80)


def train_single_seed_gpu(
    seed: int,
    hidden_dim: int,
    dense_dim: int = 64,
    episodes: int = 30,
    episode_length: int = 300,
    val_interval: int = 5,
    val_seeds: list[int] | None = None,
    early_stopping: bool = True,
    patience: int = 4,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    sequence_length: int = 10,
    burn_in: int = 0,
    use_stored_hidden: bool = True,
    regime_balanced_replay: bool = False,
    warmup_steps: int = 64,
    target_update_freq: int = 100,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9992,
    quiet: bool = False,
    train_scenarios: list[tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Trains a single LSTM-DDQN seed on CUDA with periodic held-out validation."""
    t0 = time.time()
    if val_seeds is None:
        val_seeds = [10, 20, 30]
    if train_scenarios is None:
        train_scenarios = list(get_train_scenarios().items())
    val_scenarios = get_validation_scenarios()

    env_probe = build_environment(copy.deepcopy(train_scenarios[0][1]))
    bands = env_probe.bands_hz
    num_bins = len(bands)

    encoder = TemporalObservationEncoder.create_encoder_b(num_bins=num_bins)

    torch.cuda.reset_peak_memory_stats()
    vram_start_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    sched = TorchLSTMDDQNScheduler(
        bands_hz=bands,
        encoder=encoder,
        hidden_dim=hidden_dim,
        dense_dim=dense_dim,
        sequence_length=sequence_length,
        burn_in=burn_in,
        use_stored_hidden=use_stored_hidden,
        regime_balanced_replay=regime_balanced_replay,
        learning_rate=learning_rate,
        gamma=0.95,
        replay_capacity=10000,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        target_update_frequency=target_update_freq,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        device="cuda:0",
        seed=seed,
    )
    sched.train()

    best_val_ir = -1.0
    best_weights_online: dict[str, np.ndarray] | None = None
    best_weights_target: dict[str, np.ndarray] | None = None
    best_val_metrics: dict[str, Any] = {}
    patience_counter = 0

    episode_rewards: list[float] = []
    episode_losses: list[float] = []
    val_history: list[dict[str, Any]] = []

    if not quiet:
        print(f"   [Seed {seed:3d}] Starting GPU training (H={hidden_dim}, {episodes} eps, {len(train_scenarios)} regimes)...")

    for ep in range(episodes):
        sc_name, sc_dict = train_scenarios[ep % len(train_scenarios)]
        sched.current_regime_id = ep % len(train_scenarios)
        sc_copy = copy.deepcopy(sc_dict)
        ep_seed = seed + ep * 1000
        sc_copy["simulation"]["seed"] = ep_seed
        sc_copy["simulation"]["total_time_steps"] = episode_length

        env = build_environment(sc_copy)
        env.scheduler = sched

        curr_eps = sched.epsilon
        env.reset(seed=ep_seed)
        sched.epsilon = curr_eps

        step_res = env.run(steps=episode_length)
        total_rew = sum(float(r.reward) for r in step_res)
        avg_loss = float(np.mean(sched.losses[-episode_length:])) if sched.losses else 0.0

        episode_rewards.append(total_rew)
        episode_losses.append(avg_loss)

        # Periodic Validation
        is_val_step = (ep + 1) % val_interval == 0 or (ep == episodes - 1)
        if is_val_step:
            # Export to lightweight NumPy scheduler for validation
            numpy_sched = sched.to_numpy_scheduler()
            val_eval = run_validation_battery(numpy_sched, val_scenarios, val_seeds, steps=episode_length)
            val_ir = val_eval["ir_mean"]
            val_history.append({
                "episode": ep + 1,
                "epsilon": float(sched.epsilon),
                "val_ir_mean": val_ir,
                "val_ir_std": val_eval["ir_std"],
                "val_ir_min": val_eval["ir_min"],
            })

            if not quiet:
                print(
                    f"      Ep {ep+1:2d}/{episodes:2d} | Eps: {sched.epsilon:.3f} | "
                    f"Rew: {total_rew:6.1f} | Loss: {avg_loss:.4f} | "
                    f"Val IR: {val_ir*100:5.1f}% (±{val_eval['ir_std']*100:4.1f}%)"
                )

            if val_ir > best_val_ir:
                best_val_ir = val_ir
                best_weights_online = copy.deepcopy(sched.online_net.to_numpy_weights())
                best_weights_target = copy.deepcopy(sched.target_net.to_numpy_weights())
                best_val_metrics = val_eval
                patience_counter = 0
            else:
                patience_counter += 1
                if early_stopping and patience_counter >= patience and ep >= 15:
                    if not quiet:
                        print(f"      [Early Stopping] Triggered at Ep {ep+1} (patience={patience}).")
                    break

    # Restore best validation weights
    if best_weights_online is not None and best_weights_target is not None:
        sched.online_net.from_numpy_weights(best_weights_online)
        sched.target_net.from_numpy_weights(best_weights_target)
    else:
        numpy_sched = sched.to_numpy_scheduler()
        best_val_metrics = run_validation_battery(numpy_sched, val_scenarios, val_seeds, steps=episode_length)
        best_val_ir = best_val_metrics["ir_mean"]

    sched.eval()
    sched.epsilon = 0.0

    vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    vram_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
    training_time_s = time.time() - t0

    # Training dynamics telemetry
    grad_norm_mean = float(np.mean(sched.grad_norms)) if sched.grad_norms else 0.0
    grad_norm_max = float(np.max(sched.grad_norms)) if sched.grad_norms else 0.0
    td_error_mean = float(np.mean(sched.td_errors)) if sched.td_errors else 0.0
    q_mean = float(np.mean(sched.q_means)) if sched.q_means else 0.0
    q_var = float(np.mean(sched.q_vars)) if sched.q_vars else 0.0
    h_weight_norm = float(torch.norm(sched.online_net.lstm.weight_hh_l0).item())

    return {
        "seed": seed,
        "scheduler": sched,
        "hidden_dim": hidden_dim,
        "episodes_trained": ep + 1,
        "total_steps_trained": (ep + 1) * episode_length,
        "training_time_s": training_time_s,
        "steps_per_sec": ((ep + 1) * episode_length) / max(training_time_s, 0.001),
        "best_val_ir_mean": float(best_val_ir),
        "best_val_ir_std": float(best_val_metrics.get("ir_std", 0.0)),
        "best_val_ir_min": float(best_val_metrics.get("ir_min", 0.0)),
        "best_val_det_rate": float(best_val_metrics.get("det_rate_mean", 0.0)),
        "vram_peak_mb": vram_peak_mb,
        "vram_reserved_mb": vram_reserved_mb,
        "dynamics": {
            "grad_norm_mean": grad_norm_mean,
            "grad_norm_max": grad_norm_max,
            "td_error_mean": td_error_mean,
            "q_mean": q_mean,
            "q_var": q_var,
            "h_weight_norm": h_weight_norm,
        },
        "val_history": val_history,
    }


def run_capacity_condition(
    condition_id: str,
    condition_name: str,
    hidden_dim: int,
    dense_dim: int = 64,
    seeds: list[int] = [42, 123, 456, 789],
    episodes: int = 30,
    episode_length: int = 300,
    eval_seeds: list[int] = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Executes full offline GPU training, validation, promotion, and canonical test evaluation for one capacity condition."""
    t_start = time.time()
    if output_dir is None:
        output_dir = PROJECT_ROOT / f"models/ablation/capacity/{condition_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cond_summary_path = output_dir / f"{condition_id}_condition_results.json"
    if cond_summary_path.exists():
        try:
            with open(cond_summary_path, "r") as f:
                saved_res = json.load(f)
            print(f"\n♻️  [Cached Condition] Found completed results for {condition_id.upper()} (Winner: Seed {saved_res.get('best_seed')})")
            return saved_res
        except Exception:
            pass

    train_scenarios = list(get_train_scenarios().items())

    print("\n" + "=" * 80)
    print(f"🔬 RUNNING CONDITION: {condition_id.upper()} — {condition_name}")
    print(f"   Architecture: Hidden Dim = {hidden_dim}, Dense Dim = {dense_dim} (cuDNN LSTM on CUDA)")
    print(f"   Replay: Stored Hidden = ON, Burn-in = 0, Balanced = OFF (R2 Best Mechanics)")
    print(f"   Regimes ({len(train_scenarios)}): {[s[0] for s in train_scenarios]}")
    print(f"   Training Seeds: {seeds} | Episodes: {episodes} | Steps/Ep: {episode_length}")
    print(f"   Output Checkpoint Directory: {output_dir}")
    print("=" * 80)


    candidate_records: list[dict[str, Any]] = []

    # Sequential GPU Training across seeds
    for s in seeds:
        cand_path = output_dir / f"lstm_ddqn_seed{s}.npz"
        summary_path = output_dir / f"lstm_ddqn_seed{s}_summary.json"

        if cand_path.exists() and summary_path.exists():
            with open(summary_path, "r") as f:
                cand_res = json.load(f)
            candidate_records.append(cand_res)
            print(
                f"   ♻️  [Cached Candidate] Seed {s:3d} -> Val IR: {cand_res['best_val_ir_mean']*100:5.1f}% "
                f"(±{cand_res['best_val_ir_std']*100:4.1f}%), Min: {cand_res['best_val_ir_min']*100:4.1f}% | "
                f"Speed: {cand_res.get('steps_per_sec', 0):.0f} steps/s | SHA256: {cand_res['sha256'][:16]}..."
            )
            continue

        cand_res = train_single_seed_gpu(
            seed=s,
            hidden_dim=hidden_dim,
            dense_dim=dense_dim,
            episodes=episodes,
            episode_length=episode_length,
            train_scenarios=train_scenarios,
        )
        sched = cand_res["scheduler"]

        sched.save_numpy_checkpoint(
            cand_path,
            metadata={
                "condition": condition_id,
                "condition_name": condition_name,
                "hidden_dim": hidden_dim,
                "training_seed": s,
                "episodes_trained": cand_res["episodes_trained"],
                "best_val_ir_mean": cand_res["best_val_ir_mean"],
                "best_val_ir_std": cand_res["best_val_ir_std"],
                "role": "candidate",
            },
        )

        # Mathematical Equivalence Verification
        equiv_metrics = verify_checkpoint_file_equivalence(
            checkpoint_path=cand_path,
            torch_net=sched.online_net,
            num_samples=50,
            tolerance=0.05,
        )

        cand_res["checkpoint_file"] = str(cand_path)
        cand_res["sha256"] = equiv_metrics["sha256"]
        cand_res["max_abs_diff"] = equiv_metrics["max_abs_diff"]

        # Save summary JSON (excluding non-serializable objects)
        serializable_cand = {
            "seed": cand_res["seed"],
            "hidden_dim": cand_res["hidden_dim"],
            "episodes_trained": cand_res["episodes_trained"],
            "total_steps_trained": cand_res["total_steps_trained"],
            "training_time_s": cand_res["training_time_s"],
            "steps_per_sec": cand_res["steps_per_sec"],
            "best_val_ir_mean": cand_res["best_val_ir_mean"],
            "best_val_ir_std": cand_res["best_val_ir_std"],
            "best_val_ir_min": cand_res["best_val_ir_min"],
            "best_val_det_rate": cand_res.get("best_val_det_rate", 0.0),
            "vram_peak_mb": cand_res["vram_peak_mb"],
            "vram_reserved_mb": cand_res["vram_reserved_mb"],
            "dynamics": cand_res["dynamics"],
            "checkpoint_file": str(cand_path),
            "sha256": cand_res["sha256"],
            "max_abs_diff": cand_res["max_abs_diff"],
        }
        with open(summary_path, "w") as f:
            json.dump(serializable_cand, f, indent=2)

        candidate_records.append(cand_res)

        print(
            f"   ✅ [Candidate] Seed {s:3d} -> Val IR: {cand_res['best_val_ir_mean']*100:5.1f}% "
            f"(±{cand_res['best_val_ir_std']*100:4.1f}%), Min: {cand_res['best_val_ir_min']*100:4.1f}% | "
            f"Speed: {cand_res['steps_per_sec']:.0f} steps/s | "
            f"Equiv Diff: {cand_res['max_abs_diff']:.2e} | SHA256: {cand_res['sha256'][:16]}..."
        )


    # Candidate Promotion strictly via held-out validation objective
    best_candidate, rationale = select_best_candidate(candidate_records)
    promoted_path = output_dir / f"{condition_id}_promoted.npz"
    shutil.copyfile(best_candidate["checkpoint_file"], promoted_path)

    print(f"\n⭐ PROMOTED WINNER ({condition_id.upper()}): Seed {best_candidate['seed']}")
    print(f"   Val IR: {best_candidate['best_val_ir_mean']*100:.1f}% (±{best_candidate['best_val_ir_std']*100:.1f}%)")
    print(f"   Rationale: {rationale}")
    print(f"   Promoted Checkpoint: {promoted_path}")

    # Canonical Test Suite Evaluation (untouched 8 test scenarios, 10 seeds)
    print(f"\n📊 Evaluating {condition_id.upper()} on 8 Canonical Test Scenarios (10 seeds × 300 steps)...")
    t_eval_start = time.time()
    test_eval_results = evaluate_canonical_test_suite(
        production_checkpoint_path=promoted_path,
        eval_seeds=eval_seeds,
        steps=300,
    )
    t_eval_elapsed = time.time() - t_eval_start

    # Compute overall test metrics
    v41_irs = [sc_data["v41_hybrid_ir_mean"] for sc_data in test_eval_results.values()]
    v41_stds = [sc_data["v41_hybrid_ir_std"] for sc_data in test_eval_results.values()]
    v41_dets = [sc_data["v41_det_rate_mean"] for sc_data in test_eval_results.values()]

    # Aggregate arbitrator mode counts across test scenarios
    total_modes: dict[str, int] = {}
    for sc_data in test_eval_results.values():
        for m_name, m_cnt in sc_data.get("mode_distribution", {}).items():
            total_modes[m_name] = total_modes.get(m_name, 0) + m_cnt
    total_decisions = max(sum(total_modes.values()), 1)
    mode_distribution_pct = {
        k: float(v / total_decisions * 100.0) for k, v in total_modes.items()
    }

    overall = {
        "test_ir_mean": float(np.mean(v41_irs)),
        "test_ir_std": float(np.mean(v41_stds)),
        "test_det_rate_mean": float(np.mean(v41_dets)),
    }

    t_total = time.time() - t_start
    training_time_total = sum(c["training_time_s"] for c in candidate_records)

    print(
        f"🎯 {condition_id.upper()} FINAL CANONICAL TEST RESULT: "
        f"IR = {overall['test_ir_mean']*100:.1f}% (±{overall['test_ir_std']*100:.1f}%), "
        f"Det = {overall['test_det_rate_mean']*100:.1f}% [Eval: {t_eval_elapsed:.1f}s | Train: {training_time_total:.1f}s]"
    )

    res_payload = {
        "condition_id": condition_id,
        "condition_name": condition_name,
        "hidden_dim": hidden_dim,
        "dense_dim": dense_dim,
        "training_time_s": float(training_time_total),
        "evaluation_time_s": float(t_eval_elapsed),
        "total_time_s": float(t_total),
        "best_seed": best_candidate["seed"],
        "promoted_checkpoint": str(promoted_path),
        "promoted_sha256": best_candidate["sha256"],
        "validation_metrics": {
            "val_ir_mean": float(best_candidate["best_val_ir_mean"]),
            "val_ir_std": float(best_candidate["best_val_ir_std"]),
            "val_ir_min": float(best_candidate["best_val_ir_min"]),
            "val_det_rate": float(best_candidate.get("best_val_det_rate", 0.0)),
        },
        "canonical_test_metrics": {
            "test_ir_mean": float(overall["test_ir_mean"]),
            "test_ir_std": float(overall["test_ir_std"]),
            "test_det_rate_mean": float(overall["test_det_rate_mean"]),
        },
        "mode_distribution": total_modes,
        "mode_distribution_pct": mode_distribution_pct,
        "per_scenario_results": test_eval_results,
        "best_candidate_dynamics": best_candidate.get("dynamics", {}),
        "candidates": [
            {
                "seed": c["seed"],
                "best_val_ir_mean": float(c["best_val_ir_mean"]),
                "best_val_ir_std": float(c["best_val_ir_std"]),
                "best_val_ir_min": float(c["best_val_ir_min"]),
                "episodes_trained": c["episodes_trained"],
                "training_time_s": float(c["training_time_s"]),
                "steps_per_sec": float(c["steps_per_sec"]),
                "vram_peak_mb": float(c["vram_peak_mb"]),
                "sha256": c["sha256"],
                "equiv_max_abs_diff": float(c["max_abs_diff"]),
            }
            for c in candidate_records
        ],
    }

    with open(cond_summary_path, "w") as f:
        json.dump(res_payload, f, indent=2)

    return res_payload



def main() -> int:
    print("=" * 80)
    print("🚀 SIH26055: GPU-ACCELERATED V4.1 LSTM CAPACITY ABLATION")
    print("   Comparing Hidden Dimensions: H=64 (C64) vs H=128 (C128) vs H=256 (C256)")
    print("   Training Distribution: Full 6 Diverse Regimes | Replay: R2 Best Config")
    print("=" * 80)

    # 1. Hardware & Environment Audit
    hw = get_hardware_audit()
    print_hardware_banner(hw)

    # 2. Dataset Isolation Verification
    verify_dataset_isolation()

    # 3. Experimental Matrix
    conditions = [
        ("c64", "V4.1 Baseline Capacity (H=64)", 64),
        ("c128", "V4.1 Moderate Capacity (H=128)", 128),
        ("c256", "V4.1 High Capacity (H=256)", 256),
    ]

    all_results: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": hw,
        "conditions": {},
    }

    t_suite_start = time.time()

    for cond_id, cond_name, h_dim in conditions:
        cond_res = run_capacity_condition(
            condition_id=cond_id,
            condition_name=cond_name,
            hidden_dim=h_dim,
            dense_dim=64,
            seeds=[42, 123, 456, 789],
            episodes=30,
            episode_length=300,
            eval_seeds=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        )
        all_results["conditions"][cond_id] = cond_res

    total_suite_time_s = time.time() - t_suite_start
    all_results["total_suite_time_s"] = total_suite_time_s

    # 4. Save JSON Results
    results_path = PROJECT_ROOT / "models/ablation/capacity_ablation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"📁 Full Ablation Results Saved to: {results_path}")
    print("=" * 80)

    # 5. Print Formatted Summary Tables
    print("\n" + "=" * 80)
    print("📊 RECURRENT CAPACITY ABLATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Condition':<8} | {'Hidden':<7} | {'Val IR':<14} | {'Test IR':<14} | {'Det Rate':<10} | {'Train Time':<10} | {'Winner':<6}")
    print("-" * 80)

    for cond_id in ["c64", "c128", "c256"]:
        c = all_results["conditions"][cond_id]
        v_ir = f"{c['validation_metrics']['val_ir_mean']*100:.1f}% (±{c['validation_metrics']['val_ir_std']*100:.1f}%)"
        t_ir = f"{c['canonical_test_metrics']['test_ir_mean']*100:.1f}% (±{c['canonical_test_metrics']['test_ir_std']*100:.1f}%)"
        d_rate = f"{c['canonical_test_metrics']['test_det_rate_mean']*100:.1f}%"
        t_time = f"{c['training_time_s']:.1f}s"
        winner = f"Seed {c['best_seed']}"
        print(f"{cond_id.upper():<8} | {c['hidden_dim']:<7} | {v_ir:<14} | {t_ir:<14} | {d_rate:<10} | {t_time:<10} | {winner:<6}")

    print("-" * 80)

    # Speedup Comparison vs CPU Baseline
    print("\n" + "=" * 80)
    print("⚡ COMPUTE ACCELERATION & SPEEDUP ANALYSIS")
    print("=" * 80)
    cpu_ref_time_s = 4194.0  # 69.9 minutes from previous CPU multi-process replay ablation
    gpu_total_train_s = sum(all_results["conditions"][cid]["training_time_s"] for cid in ["c64", "c128", "c256"])
    speedup = cpu_ref_time_s / max(gpu_total_train_s, 1.0)
    print(f"   Historical NumPy CPU Training Time (4 seeds): ~{cpu_ref_time_s/60:.1f} min ({cpu_ref_time_s:.0f}s)")
    print(f"   GPU Training Time across C64 + C128 + C256:  {gpu_total_train_s/60:.2f} min ({gpu_total_train_s:.1f}s)")
    print(f"   Aggregate GPU Training Speedup:               {speedup:.1f}x FASTER")
    print(f"   Total End-to-End Suite Time (Train + Test):   {total_suite_time_s/60:.2f} min ({total_suite_time_s:.1f}s)")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
