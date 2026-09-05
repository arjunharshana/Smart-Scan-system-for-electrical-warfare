"""benchmarks/train_v4_1.py
Official Offline Training, Validation, and Checkpoint-Selection Pipeline for V4.1.

Lifecycle:
    Diverse Training Scenarios
                ↓
    Offline Multi-Seed Training (LSTM-DDQN)
                ↓
    Periodic Held-Out Validation (V4.1 Hybrid)
                ↓
    Model Selection (Best Validation IR & Stability)
                ↓
    Promotion to models/v4_1/production_checkpoint.npz
                ↓
    Manifest Generation (models/v4_1/training_manifest.json)
                ↓
    Untouched Canonical Test Evaluation
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from benchmarks.scenarios_v4_1 import (
    SPECTRUM_CFG,
    get_canonical_test_scenarios,
    get_train_scenarios,
    get_validation_scenarios,
    verify_dataset_isolation,
)
from rf_environment.domain.action import ScanAction
from rf_environment.domain.state import SchedulerObservation
from rf_environment.environment.builder import build_environment
from rf_environment.scheduler.context_aware import ContextAwareScheduler
from rf_environment.scheduler.hybrid.hybrid_v41 import LSTMHybridScheduler
from rf_environment.scheduler.rl.checkpoint import compute_weights_sha256, save_checkpoint
from rf_environment.scheduler.rl.lstm_ddqn_scheduler import LSTMDDQNScheduler
from rf_environment.scheduler.rl.temporal_encoder import TemporalObservationEncoder
from rf_environment.util.seeding import derive_seed


def evaluate_hybrid_on_scenario(
    scenario_dict: dict[str, Any],
    lstm_scheduler: LSTMDDQNScheduler,
    seeds: list[int],
    steps: int = 300,
) -> dict[str, Any]:
    """Evaluates V4.1 Hybrid (Context-Aware + LSTM-DDQN) on a scenario across seeds.
    Strictly uses eval mode (epsilon=0.0, zero gradient updates, hidden state reset).
    """
    irs: list[float] = []
    det_rates: list[float] = []
    scan_effs: list[float] = []

    for s in seeds:
        sc_copy = copy.deepcopy(scenario_dict)
        sc_copy["simulation"]["seed"] = s
        sc_copy["simulation"]["total_time_steps"] = steps

        env = build_environment(sc_copy)
        bands = env.bands_hz

        # Clone and freeze LSTM scheduler
        trained_lstm = copy.deepcopy(lstm_scheduler)
        trained_lstm.bands_hz = bands
        trained_lstm.eval()
        trained_lstm.epsilon = 0.0

        hybrid = LSTMHybridScheduler(
            bands_hz=bands,
            lstm_ddqn=trained_lstm,
            seed=derive_seed(s, "eval_hybrid"),
        )
        hybrid.eval()
        env.scheduler = hybrid
        hybrid.reset()

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
    lstm_scheduler: LSTMDDQNScheduler,
    val_scenarios: dict[str, dict[str, Any]],
    val_seeds: list[int],
    steps: int = 300,
) -> dict[str, Any]:
    """Evaluates the candidate scheduler across all held-out validation scenarios."""
    sc_results: dict[str, Any] = {}
    all_irs: list[float] = []
    all_dets: list[float] = []

    for sc_name, sc_cfg in val_scenarios.items():
        res = evaluate_hybrid_on_scenario(sc_cfg, lstm_scheduler, val_seeds, steps=steps)
        sc_results[sc_name] = res
        all_irs.append(res["ir_mean"])
        all_dets.append(res["det_rate_mean"])

    return {
        "ir_mean": float(np.mean(all_irs)),
        "ir_std": float(np.std(all_irs)),
        "ir_min": float(np.min(all_irs)),
        "det_rate_mean": float(np.mean(all_dets)),
        "scenario_metrics": sc_results,
    }


def train_single_seed(
    seed: int,
    episodes: int = 25,
    episode_length: int = 300,
    val_interval: int = 5,
    val_seeds: list[int] | None = None,
    early_stopping: bool = True,
    patience: int = 4,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    sequence_length: int = 10,
    warmup_steps: int = 64,
    target_update_freq: int = 100,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.9992,
    quiet: bool = False,
    train_scenarios: list[tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Trains an independent candidate LSTM-DDQN model for one seed with held-out validation."""
    if val_seeds is None:
        val_seeds = [10, 20, 30]

    if train_scenarios is None:
        train_scenarios = list(get_train_scenarios().items())
    val_scenarios = get_validation_scenarios()

    # Determine frequency bands from standard spectrum
    env_probe = build_environment(copy.deepcopy(train_scenarios[0][1]))
    bands = env_probe.bands_hz

    encoder = TemporalObservationEncoder.create_encoder_b(num_bins=len(bands))
    lstm_sched = LSTMDDQNScheduler(
        bands_hz=bands,
        encoder=encoder,
        hidden_dim=64,
        dense_dim=64,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        gamma=0.95,
        replay_capacity=10000,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        target_update_frequency=target_update_freq,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    lstm_sched.train()

    best_val_ir = -1.0
    best_weights_online: dict[str, np.ndarray] | None = None
    best_weights_target: dict[str, np.ndarray] | None = None
    best_val_metrics: dict[str, Any] = {}
    patience_counter = 0

    episode_rewards: list[float] = []
    episode_losses: list[float] = []
    val_history: list[dict[str, Any]] = []

    if not quiet:
        print(f"\n[Seed {seed}] Starting training ({episodes} episodes, {len(train_scenarios)} train scenarios)...")

    for ep in range(episodes):
        sc_name, sc_dict = train_scenarios[ep % len(train_scenarios)]
        sc_copy = copy.deepcopy(sc_dict)
        ep_seed = seed + ep * 1000
        sc_copy["simulation"]["seed"] = ep_seed
        sc_copy["simulation"]["total_time_steps"] = episode_length

        env = build_environment(sc_copy)
        env.scheduler = lstm_sched

        # Preserve continuous decaying epsilon across episode resets
        curr_eps = lstm_sched.epsilon
        env.reset(seed=ep_seed)
        lstm_sched.epsilon = curr_eps

        step_res = env.run(steps=episode_length)
        total_rew = sum(float(r.reward) for r in step_res)
        avg_loss = float(np.mean(lstm_sched.losses[-episode_length:])) if lstm_sched.losses else 0.0

        episode_rewards.append(total_rew)
        episode_losses.append(avg_loss)

        # Periodic Validation
        is_val_step = (ep + 1) % val_interval == 0 or (ep == episodes - 1)
        if is_val_step:
            val_eval = run_validation_battery(lstm_sched, val_scenarios, val_seeds, steps=episode_length)
            val_ir = val_eval["ir_mean"]
            val_history.append({
                "episode": ep + 1,
                "epsilon": float(lstm_sched.epsilon),
                "val_ir_mean": val_ir,
                "val_ir_std": val_eval["ir_std"],
                "val_ir_min": val_eval["ir_min"],
            })

            if not quiet:
                print(
                    f"   Ep {ep+1:2d}/{episodes:2d} | Eps: {lstm_sched.epsilon:.3f} | "
                    f"Rew: {total_rew:6.1f} | Loss: {avg_loss:.4f} | "
                    f"Val IR: {val_ir*100:5.1f}% (±{val_eval['ir_std']*100:4.1f}%)"
                )

            if val_ir > best_val_ir:
                best_val_ir = val_ir
                best_weights_online = copy.deepcopy(lstm_sched.online_net.get_weights_dict())
                best_weights_target = copy.deepcopy(lstm_sched.target_net.get_weights_dict())
                best_val_metrics = val_eval
                patience_counter = 0
            else:
                patience_counter += 1
                if early_stopping and patience_counter >= patience and ep >= 15:
                    if not quiet:
                        print(f"   [Early Stopping] No validation improvement for {patience} checks. Stopping at Ep {ep+1}.")
                    break

    # Restore best validation weights
    if best_weights_online is not None and best_weights_target is not None:
        lstm_sched.online_net.load_weights_dict(best_weights_online)
        lstm_sched.target_net.load_weights_dict(best_weights_target)
    else:
        best_val_metrics = run_validation_battery(lstm_sched, val_scenarios, val_seeds, steps=episode_length)
        best_val_ir = best_val_metrics["ir_mean"]

    lstm_sched.eval()
    lstm_sched.epsilon = 0.0

    return {
        "seed": seed,
        "scheduler": lstm_sched,
        "episodes_trained": ep + 1,
        "final_epsilon": float(lstm_sched.epsilon),
        "best_val_ir_mean": float(best_val_ir),
        "best_val_ir_std": float(best_val_metrics.get("ir_std", 0.0)),
        "best_val_ir_min": float(best_val_metrics.get("ir_min", 0.0)),
        "best_val_det_rate": float(best_val_metrics.get("det_rate_mean", 0.0)),
        "val_scenario_metrics": best_val_metrics.get("scenario_metrics", {}),
        "val_history": val_history,
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
    }


def select_best_candidate(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Selects the single best candidate checkpoint based on validation IR and stability."""
    if not candidates:
        raise ValueError("Cannot select from empty candidate list")

    # Score rule: primary is mean validation IR; penalty for high scenario variance; reject severe collapse
    scored: list[tuple[float, dict[str, Any], str]] = []

    for c in candidates:
        ir_m = c["best_val_ir_mean"]
        ir_s = c["best_val_ir_std"]
        ir_min = c["best_val_ir_min"]

        # If min scenario IR < 0.10, penalize heavily as inconsistent
        collapse_penalty = 0.20 if ir_min < 0.10 else 0.0
        score = ir_m - 0.5 * ir_s - collapse_penalty

        rationale = (
            f"Val IR: {ir_m*100:.1f}%, Std: {ir_s*100:.1f}%, Min: {ir_min*100:.1f}% "
            f"(Score: {score*100:.2f})"
        )
        scored.append((score, c, rationale))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cand, best_rat = scored[0]

    full_reason = (
        f"Selected Seed {best_cand['seed']} with highest validation composite score ({best_score*100:.2f}). "
        f"Validation Interception Rate: {best_cand['best_val_ir_mean']*100:.2f}% "
        f"(±{best_cand['best_val_ir_std']*100:.2f}%), Min Scenario IR: {best_cand['best_val_ir_min']*100:.2f}%. "
        f"Demonstrated superior stability across diverse held-out validation scenarios without collapse."
    )

    return best_cand, full_reason


def evaluate_canonical_test_suite(
    production_checkpoint_path: Path,
    eval_seeds: list[int] | None = None,
    steps: int = 300,
) -> dict[str, Any]:
    """Runs untouched canonical test benchmark evaluation with the promoted production checkpoint."""
    if eval_seeds is None:
        eval_seeds = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]

    test_scenarios = get_canonical_test_scenarios()
    results: dict[str, Any] = {}

    print("\n" + "=" * 80)
    print("📊 UNTOUCHED CANONICAL TEST BENCHMARK EVALUATION (FROZEN PRODUCTION MODEL)")
    print("=" * 80)
    print(f"Checkpoint: {production_checkpoint_path}")
    print(f"Evaluation Seeds: {eval_seeds} ({len(eval_seeds)} seeds × 300 steps)")
    print("-" * 80)
    print(f"{'Scenario':<25} | {'CA Baseline':<12} | {'V4.1 Hybrid':<12} | {'Detection %':<12} | {'Mode Distribution'}")
    print("-" * 80)

    for sc_name, sc_cfg in test_scenarios.items():
        # 1. Context-Aware Baseline
        ca_irs = []
        for s in eval_seeds:
            sc_copy = copy.deepcopy(sc_cfg)
            sc_copy["simulation"]["seed"] = s
            sc_copy["simulation"]["total_time_steps"] = steps
            env_ca = build_environment(sc_copy, scheduler_name="context_aware")
            env_ca.run(steps=steps)
            ca_irs.append(env_ca.metrics.snapshot(steps).interception_ratio)
        ca_ir_mean = float(np.mean(ca_irs))

        # 2. V4.1 Hybrid with production checkpoint
        v41_irs, v41_dets, v41_effs = [], [], []
        mode_counts_total: dict[str, int] = {}

        for s in eval_seeds:
            sc_copy = copy.deepcopy(sc_cfg)
            sc_copy["simulation"]["seed"] = s
            sc_copy["simulation"]["total_time_steps"] = steps

            env_v41 = build_environment(
                sc_copy,
                scheduler_name="hybrid_v41",
                checkpoint_path=str(production_checkpoint_path),
                require_checkpoint=True,
            )
            env_v41.scheduler.eval()
            env_v41.scheduler.epsilon = 0.0

            env_v41.run(steps=steps)
            snap = env_v41.metrics.snapshot(steps)

            v41_irs.append(float(snap.interception_ratio))
            v41_dets.append(float(snap.probability_of_detection))

            # Accumulate mode counts
            if hasattr(env_v41.scheduler, "mode_counts"):
                for m_k, m_v in env_v41.scheduler.mode_counts.items():
                    mode_counts_total[m_k] = mode_counts_total.get(m_k, 0) + m_v

        v41_ir_mean = float(np.mean(v41_irs))
        v41_ir_std = float(np.std(v41_irs))
        v41_det_mean = float(np.mean(v41_dets))

        # Format mode shares
        tot_dec = max(sum(mode_counts_total.values()), 1)
        p_exploit = mode_counts_total.get("DDQN_EXPLOIT", 0) / tot_dec * 100.0
        p_adapt = mode_counts_total.get("CA_ADAPT", 0) / tot_dec * 100.0
        p_blend = mode_counts_total.get("BLENDED", 0) / tot_dec * 100.0
        mode_str = f"{p_exploit:4.0f}% Exploit / {p_adapt:3.0f}% Adapt / {p_blend:3.0f}% Blend"

        print(
            f"{sc_name:<25} | {ca_ir_mean*100:10.1f}% | {v41_ir_mean*100:10.1f}% | "
            f"{v41_det_mean*100:10.1f}% | {mode_str}"
        )

        results[sc_name] = {
            "ca_ir_mean": ca_ir_mean,
            "v41_hybrid_ir_mean": v41_ir_mean,
            "v41_hybrid_ir_std": v41_ir_std,
            "v41_det_rate_mean": v41_det_mean,
            "per_seed_ir": v41_irs,
            "mode_distribution": mode_counts_total,
        }

    print("-" * 80)
    return results


def run_pipeline(
    seeds: list[int] | None = None,
    episodes: int = 25,
    episode_length: int = 300,
    val_interval: int = 5,
    checkpoint_dir: str | Path = "models/v4_1",
    learning_rate: float = 0.001,
    batch_size: int = 32,
    sequence_length: int = 10,
    warmup_steps: int = 64,
    target_update_freq: int = 100,
    epsilon_decay: float = 0.9992,
    patience: int = 4,
    early_stopping: bool = True,
    evaluate_test: bool = True,
    quiet: bool = False,
    train_scenarios: list[tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Executes the full offline training, validation, checkpoint selection,
    promotion, and test evaluation pipeline.
    """
    t_start = time.time()
    if seeds is None:
        seeds = [42, 123, 456, 789]

    # Verify zero data leakage between sets
    verify_dataset_isolation()

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 SIH26055 — V4.1 OFFLINE TRAINING & VALIDATION PIPELINE")
    print("=" * 80)
    print(f"Training Seeds:       {seeds}")
    print(f"Episodes Per Seed:    {episodes} (Episode Length: {episode_length} steps)")
    print(f"Validation Interval:  Every {val_interval} episodes")
    print(f"Output Directory:     {ckpt_dir}")
    print(f"Early Stopping:       {early_stopping} (Patience: {patience})")
    print("=" * 80)

    candidate_records: list[dict[str, Any]] = []

    for s in seeds:
        res = train_single_seed(
            seed=s,
            episodes=episodes,
            episode_length=episode_length,
            val_interval=val_interval,
            val_seeds=[10, 20, 30],
            early_stopping=early_stopping,
            patience=patience,
            learning_rate=learning_rate,
            batch_size=batch_size,
            sequence_length=sequence_length,
            warmup_steps=warmup_steps,
            target_update_freq=target_update_freq,
            epsilon_decay=epsilon_decay,
            quiet=quiet,
            train_scenarios=train_scenarios,
        )

        sched = res["scheduler"]
        cand_path = ckpt_dir / f"lstm_ddqn_seed{s}.npz"
        sched.save_checkpoint(
            cand_path,
            metadata={
                "training_seed": s,
                "episodes_trained": res["episodes_trained"],
                "best_val_ir_mean": res["best_val_ir_mean"],
                "best_val_ir_std": res["best_val_ir_std"],
                "role": "candidate",
            },
        )
        cand_hash = sched.checkpoint_sha256

        res["checkpoint_file"] = str(cand_path)
        res["sha256"] = cand_hash
        candidate_records.append(res)

        print(
            f"✅ [Candidate Ready] Seed {s:3d} -> {cand_path.name} | "
            f"Val IR: {res['best_val_ir_mean']*100:5.1f}% (±{res['best_val_ir_std']*100:4.1f}%) | "
            f"SHA256: {cand_hash[:16]}..."
        )

    # 2. Select Best Model Based on Validation Performance
    print("\n" + "=" * 80)
    print("🏆 MODEL SELECTION FROM HELD-OUT VALIDATION BATTERY")
    print("=" * 80)

    best_candidate, selection_reason = select_best_candidate(candidate_records)
    print(f"Selected Candidate: Seed {best_candidate['seed']}")
    print(f"Selection Reason:   {selection_reason}")

    # 3. Promote to Production Checkpoint
    prod_path = ckpt_dir / "production_checkpoint.npz"
    shutil.copyfile(best_candidate["checkpoint_file"], prod_path)
    print(f"Promoted Checkpoint: {prod_path} (SHA256: {best_candidate['sha256']})")

    # 4. Generate Audit Manifest
    manifest_payload: dict[str, Any] = {
        "architecture": "v4.1",
        "pipeline": "offline_training_and_validation",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_seeds": seeds,
        "hyperparameters": {
            "episodes": episodes,
            "episode_length": episode_length,
            "validation_interval": val_interval,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "warmup_steps": warmup_steps,
            "target_update_freq": target_update_freq,
            "epsilon_decay": epsilon_decay,
            "early_stopping": early_stopping,
            "patience": patience,
        },
        "scenarios": {
            "train": list(get_train_scenarios().keys()),
            "validation": list(get_validation_scenarios().keys()),
            "test": list(get_canonical_test_scenarios().keys()),
        },
        "candidates": [
            {
                "seed": c["seed"],
                "checkpoint_file": c["checkpoint_file"],
                "sha256": c["sha256"],
                "episodes_trained": c["episodes_trained"],
                "validation_ir_mean": c["best_val_ir_mean"],
                "validation_ir_std": c["best_val_ir_std"],
                "validation_ir_min": c["best_val_ir_min"],
                "validation_det_rate": c["best_val_det_rate"],
            }
            for c in candidate_records
        ],
        "selected_checkpoint": {
            "seed": best_candidate["seed"],
            "source_checkpoint": best_candidate["checkpoint_file"],
            "production_checkpoint": str(prod_path),
            "sha256": best_candidate["sha256"],
            "validation_ir_mean": best_candidate["best_val_ir_mean"],
            "validation_ir_std": best_candidate["best_val_ir_std"],
            "selection_reason": selection_reason,
        },
    }

    manifest_path = ckpt_dir / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    print(f"Manifest Generated:  {manifest_path}")

    # 5. Untouched Canonical Test Evaluation
    test_results = {}
    if evaluate_test:
        test_results = evaluate_canonical_test_suite(prod_path)
        test_results_path = ckpt_dir / "canonical_test_results.json"
        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump(test_results, f, indent=2)
        print(f"\nCanonical Test Results Saved: {test_results_path}")

    elapsed = time.time() - t_start
    print(f"\n✨ Pipeline Execution Completed Successfully in {elapsed:.1f}s.")

    return {
        "manifest": manifest_payload,
        "best_candidate": best_candidate,
        "test_results": test_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V4.1 Offline Training, Validation, and Model Selection Pipeline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789], help="Training random seeds")
    parser.add_argument("--episodes", type=int, default=25, help="Number of training episodes per seed")
    parser.add_argument("--episode-length", type=int, default=300, help="Steps per training episode")
    parser.add_argument("--validation-interval", type=int, default=5, help="Validate every N episodes")
    parser.add_argument("--checkpoint-dir", type=str, default="models/v4_1", help="Target checkpoint directory")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Sequence batch size")
    parser.add_argument("--sequence-length", type=int, default=10, help="Recurrent BPTT sequence length")
    parser.add_argument("--warmup-steps", type=int, default=64, help="Replay warmup transitions")
    parser.add_argument("--target-update-freq", type=int, default=100, help="Target network sync frequency")
    parser.add_argument("--epsilon-decay", type=float, default=0.9992, help="Epsilon decay rate per step")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience in validation cycles")
    parser.add_argument("--no-early-stopping", dest="early_stopping", action="store_false", help="Disable early stopping")
    parser.add_argument("--no-test-eval", dest="evaluate_test", action="store_false", help="Skip canonical test evaluation")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose progress logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        seeds=args.seeds,
        episodes=args.episodes,
        episode_length=args.episode_length,
        val_interval=args.validation_interval,
        checkpoint_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        warmup_steps=args.warmup_steps,
        target_update_freq=args.target_update_freq,
        epsilon_decay=args.epsilon_decay,
        patience=args.patience,
        early_stopping=args.early_stopping,
        evaluate_test=args.evaluate_test,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
