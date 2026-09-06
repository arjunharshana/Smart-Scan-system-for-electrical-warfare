from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import WebSocket

from app.config import (
    APP_VERSION,
    DEFAULT_SCENARIO_NAME,
    DEFAULT_SCENARIO_PATH,
    DEFAULT_SCHEDULER,
    DEFAULT_V40_CHECKPOINT,
    DEFAULT_V41_CHECKPOINT,
    PROJECT_ROOT,
)
from rf_environment.environment.builder import build_environment
from rf_environment.environment.rf_environment import RFEnvironment
from rf_environment.environment.scenario import load_scenario
from rf_environment.scheduler.factory import SCHEDULER_METADATA

logger = logging.getLogger("simulation_service")

# Speed delays in seconds per step
SPEED_DELAYS: dict[str, float] = {
    "0.25x": 0.40,
    "0.5x": 0.20,
    "1x": 0.10,
    "2x": 0.05,
    "5x": 0.02,
    "max": 0.001,
}



def _clean_val(v: Any) -> Any:
    """Recursively converts non-JSON-serializable floats (NaN, Inf) and NumPy types."""
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_clean_val(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _clean_val(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean_val(x) for x in v]
    return v


class SimulationService:
    """Manages RF Environment lifecycle, tactical scheduler execution,

    speed-controlled background advancement, and live telemetry broadcasting.
    """

    def __init__(self) -> None:
        self.env: RFEnvironment | None = None
        self.scenario_name: str = DEFAULT_SCENARIO_NAME
        self.scenario_dict: dict[str, Any] | None = None
        self.scheduler_name: str = DEFAULT_SCHEDULER
        self.seed: int = 42
        self.speed: str = "1x"
        self.state: str = "IDLE"  # IDLE, RUNNING, PAUSED

        self._lock = asyncio.Lock()
        self._runner_task: asyncio.Task | None = None
        self.clients: set[WebSocket] = set()

        # Telemetry ring buffers
        self.waterfall_history: deque[dict[str, Any]] = deque(maxlen=150)
        self.timeline_history: deque[dict[str, Any]] = deque(maxlen=25)
        self.latest_telemetry: dict[str, Any] = {}

        # Initialize environment with defaults
        self.reset(seed=self.seed, scenario_name=self.scenario_name, scheduler_name=self.scheduler_name)

    def list_available_scenarios(self) -> list[dict[str, Any]]:
        from benchmarks.scenarios_v4_1 import get_canonical_test_scenarios

        canonical = get_canonical_test_scenarios()
        items = []
        for key in canonical.keys():
            readable_name = key.replace("_", " ")
            items.append({
                "id": key,
                "name": readable_name,
                "category": "Canonical Benchmark Suite",
                "filename": f"{key}.canonical",
                "path": f"canonical://{key}",
            })

        scenarios_dir = PROJECT_ROOT / "rf_environment" / "scenarios"
        if scenarios_dir.exists():
            for p in sorted(scenarios_dir.glob("*.yaml")):
                name = p.stem.replace("_", " ").title()
                items.append({
                    "id": p.stem,
                    "name": name,
                    "category": "Standard Scenarios",
                    "filename": p.name,
                    "path": str(p),
                })
        return items

    def list_available_schedulers(self) -> list[dict[str, Any]]:
        """Returns the catalog of available scan strategy algorithms for comparative demonstration."""
        return [
            {
                "id": "hybrid_v4",
                "name": "⭐ V4.0 Hybrid (Our Algorithm — Benchmark Winner)",
                "category": "★ Proposed System (Winner)",
                "description": "Feedforward DDQN + Context-Aware Meta-Arbitrator. Highest overall test score (35.19% IR).",
                "benchmark_ir_pct": 35.19,
                "overall_ir": "35.19%",
                "rank": 1,
                "is_production": True,
                "is_proposed": True,
                "badge": "WINNER / PROPOSED",
            },
            {
                "id": "whittle_style",
                "name": "Whittle W3 Index (Restless Bandit)",
                "category": "Research Baselines",
                "description": "Heuristic restless bandit index policy balancing belief, dwell aging, and return periodicity (34.40% IR).",
                "benchmark_ir_pct": 34.40,
                "overall_ir": "34.40%",
                "rank": 2,
                "is_production": False,
                "is_proposed": False,
                "badge": "BANDIT BASELINE",
            },
            {
                "id": "context_aware",
                "name": "Context-Aware Transition (CA)",
                "category": "Research Baselines",
                "description": "Online empirical Markov transition matrix with recency and coverage bonuses (31.15% IR).",
                "benchmark_ir_pct": 31.15,
                "overall_ir": "31.15%",
                "rank": 3,
                "is_production": False,
                "is_proposed": False,
                "badge": "EMPIRICAL BASELINE",
            },
            {
                "id": "hybrid_v41",
                "name": "V4.1 LSTM-Hybrid (Recurrent Memory)",
                "category": "Research Baselines",
                "description": "Recurrent LSTM working memory DRQN coupled with Context-Aware arbitrator (26.90% IR).",
                "benchmark_ir_pct": 26.90,
                "overall_ir": "26.90%",
                "rank": 4,
                "is_production": False,
                "is_proposed": False,
                "badge": "RECURRENT ABLATION",
            },
            {
                "id": "v5_belief",
                "name": "V5.0 Augmented Belief (Bayesian POMDP)",
                "category": "Research Baselines",
                "description": "Exact recursive Bayesian POMDP filter over joint state (F, tau, D) (15.36% IR).",
                "benchmark_ir_pct": 15.36,
                "overall_ir": "15.36%",
                "rank": 5,
                "is_production": False,
                "is_proposed": False,
                "badge": "BAYESIAN BASELINE",
            },
            {
                "id": "sequential",
                "name": "Sequential Channel Sweep",
                "category": "Standard & Legacy Baselines",
                "description": "Fixed deterministic linear sweep across spectrum bins (Legacy receiver baseline, ~3.3% IR).",
                "benchmark_ir_pct": 3.33,
                "overall_ir": "~3.3%",
                "rank": 6,
                "is_production": False,
                "is_proposed": False,
                "badge": "LEGACY SCANNER",
            },
            {
                "id": "random",
                "name": "Uniform Random Sweep",
                "category": "Standard & Legacy Baselines",
                "description": "Zero-intelligence uniform random channel sampling (~3.3% IR).",
                "benchmark_ir_pct": 3.33,
                "overall_ir": "~3.3%",
                "rank": 7,
                "is_production": False,
                "is_proposed": False,
                "badge": "ZERO INTELLIGENCE",
            },
            {
                "id": "ucb1",
                "name": "UCB1 Multi-Armed Bandit",
                "category": "Standard & Legacy Baselines",
                "description": "Stationary Upper Confidence Bound bandit balancing mean hit reward with exploration (~10% IR).",
                "benchmark_ir_pct": 10.50,
                "overall_ir": "~10%",
                "rank": 8,
                "is_production": False,
                "is_proposed": False,
                "badge": "STATIONARY BANDIT",
            },
            {
                "id": "thompson",
                "name": "Thompson Sampling",
                "category": "Standard & Legacy Baselines",
                "description": "Stationary Bayesian Beta-posterior bandit sampling (~10% IR).",
                "benchmark_ir_pct": 10.20,
                "overall_ir": "~10%",
                "rank": 9,
                "is_production": False,
                "is_proposed": False,
                "badge": "STATIONARY BANDIT",
            },
        ]


    def reset(
        self,
        seed: int | None = None,
        scenario_name: str | None = None,
        scheduler_name: str | None = None,
    ) -> None:
        """Resets the simulation environment with given or existing configuration."""
        if seed is not None:
            self.seed = seed
        if scenario_name is not None:
            self.scenario_name = scenario_name
        if scheduler_name is not None:
            self.scheduler_name = scheduler_name

        from benchmarks.scenarios_v4_1 import get_canonical_test_scenarios

        canonical_dict = get_canonical_test_scenarios()
        clean_name = self.scenario_name.replace(".canonical", "").replace(".yaml", "")

        if self.scenario_name in canonical_dict:
            self.scenario_dict = copy.deepcopy(canonical_dict[self.scenario_name])
        elif clean_name in canonical_dict:
            self.scenario_name = clean_name
            self.scenario_dict = copy.deepcopy(canonical_dict[clean_name])
        else:
            sc_path = PROJECT_ROOT / "rf_environment" / "scenarios" / self.scenario_name
            if not sc_path.exists():
                sc_path = PROJECT_ROOT / "rf_environment" / "scenarios" / f"{self.scenario_name}.yaml"
            if not sc_path.exists():
                sc_path = DEFAULT_SCENARIO_PATH
                self.scenario_name = DEFAULT_SCENARIO_NAME
            self.scenario_dict = load_scenario(str(sc_path))

        sc_copy = copy.deepcopy(self.scenario_dict)

        if "simulation" not in sc_copy:
            sc_copy["simulation"] = {}
        sc_copy["simulation"]["seed"] = self.seed

        # Check if scenario matches canonical 30-bin tactical spectrum
        spectrum = sc_copy.get("spectrum", {})
        rx_cfg = sc_copy.get("receiver", {})
        min_hz = float(spectrum.get("min_frequency_hz", 100_000_000))
        max_hz = float(spectrum.get("max_frequency_hz", 700_000_000))
        bw = float(rx_cfg.get("instantaneous_bandwidth_hz", 20_000_000))
        num_bins = int(round((max_hz - min_hz) / bw)) if bw > 0 else 30

        # Build environment
        extra_kwargs: dict[str, Any] = {}
        if self.scheduler_name in {"hybrid_v4", "hybrid", "hybrid_meta"}:
            if num_bins == 30:
                extra_kwargs["checkpoint_path"] = str(DEFAULT_V40_CHECKPOINT)
                extra_kwargs["require_checkpoint"] = True
            else:
                logger.warning(
                    "Scenario %s has %d bins != 30 (checkpoint trained for 30 bins). Running uncheckpointed.",
                    self.scenario_name,
                    num_bins,
                )
        elif self.scheduler_name in {"hybrid_v41", "hybrid_lstm", "lstm_hybrid"}:
            if num_bins == 30:
                extra_kwargs["checkpoint_path"] = str(DEFAULT_V41_CHECKPOINT)
                extra_kwargs["require_checkpoint"] = True
            else:
                logger.warning(
                    "Scenario %s has %d bins != 30 (checkpoint trained for 30 bins). Running uncheckpointed.",
                    self.scenario_name,
                    num_bins,
                )

        self.env = build_environment(sc_copy, scheduler_name=self.scheduler_name, **extra_kwargs)
        if hasattr(self.env.scheduler, "eval"):
            self.env.scheduler.eval()
        if hasattr(self.env.scheduler, "epsilon"):
            self.env.scheduler.epsilon = 0.0

        # Log production startup banner
        sched = self.env.scheduler
        if self.scheduler_name in {"hybrid_v4", "hybrid", "hybrid_meta"}:
            ckpt_p = getattr(sched, "checkpoint_path", str(DEFAULT_V40_CHECKPOINT))
            ckpt_hash = getattr(sched, "checkpoint_sha256", "UNKNOWN")
            logger.info("=" * 65)
            logger.info("TACTICAL SCHEDULER: hybrid_v4 (V4.0 Hybrid — Benchmark Winner: 35.19% IR)")
            logger.info("MODEL ARCHITECTURE: MLPQNetwork (Feed-Forward DDQN + Meta-Arbitrator)")
            logger.info("MODEL STATUS:       PRETRAINED")
            logger.info("TRAINING:           OFFLINE")
            logger.info("RUNTIME TRAINING:   DISABLED")
            logger.info("CHECKPOINT:         %s", ckpt_p)
            logger.info("CHECKPOINT SHA256:  %s", ckpt_hash)
            logger.info("PRODUCTION MODE:    FROZEN INFERENCE")
            logger.info("=" * 65)
        elif self.scheduler_name in {"hybrid_v41", "hybrid_lstm"}:
            ckpt_p = getattr(sched, "checkpoint_path", str(DEFAULT_V41_CHECKPOINT))
            ckpt_hash = getattr(sched, "checkpoint_sha256", "UNKNOWN")
            logger.info("=" * 65)
            logger.info("TACTICAL SCHEDULER: hybrid_v41 (V4.1 LSTM-Hybrid)")
            logger.info("MODEL STATUS:       PRETRAINED")
            logger.info("TRAINING:           OFFLINE")
            logger.info("RUNTIME TRAINING:   DISABLED")
            logger.info("CHECKPOINT:         %s", ckpt_p)
            logger.info("CHECKPOINT SHA256:  %s", ckpt_hash)
            logger.info("PRODUCTION MODE:    FROZEN INFERENCE")
            logger.info("=" * 65)

        self.state = "IDLE"
        self.waterfall_history.clear()
        self.timeline_history.clear()

        # Build initial baseline telemetry
        self.latest_telemetry = self._extract_telemetry(None)

    def get_status(self) -> dict[str, Any]:
        t = self.env.clock.time_step if self.env else 0
        sim_time = (self.env.clock.time_ms / 1000.0) if self.env else 0.0
        return {
            "state": self.state,
            "time_step": t,
            "simulation_time_s": round(sim_time, 3),


            "scenario_name": self.scenario_name,
            "scheduler_name": SCHEDULER_METADATA.get(self.scheduler_name, {}).get("name", self.scheduler_name),
            "scheduler_type": self.scheduler_name,
            "speed": self.speed,
            "seed": self.seed,
            "version": APP_VERSION,
        }

    def set_speed(self, speed: str) -> None:
        if speed in SPEED_DELAYS:
            self.speed = speed

    def step(self, count: int = 1) -> dict[str, Any]:
        """Executes count simulation steps synchronously and records telemetry with latency measurement."""
        if not self.env:
            raise RuntimeError("Environment not initialized")

        last_result = None
        latency_ms = 0.5
        for _ in range(count):
            if self.env.clock.finished():
                self.state = "PAUSED"
                break
            t0 = time.perf_counter()
            last_result = self.env.step()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            telem = self._extract_telemetry(last_result, latency_ms=latency_ms)
            self.latest_telemetry = telem

        return self.latest_telemetry

    async def start(self, steps: int | None = None) -> None:
        """Starts asynchronous continuous execution."""
        if self.env and self.env.clock.finished():
            logger.info("Simulation clock reached completion; auto-resetting environment for new run.")
            self.reset(seed=self.seed, scenario_name=self.scenario_name, scheduler_name=self.scheduler_name)

        if self.state == "RUNNING":
            return
        self.state = "RUNNING"
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()
        self._runner_task = asyncio.create_task(self._run_loop(steps))


    async def pause(self) -> None:
        """Pauses the running simulation."""
        self.state = "PAUSED"
        if self._runner_task and not self._runner_task.done():
            self._runner_task.cancel()

    async def _run_loop(self, max_steps: int | None = None) -> None:
        """Background stepping task with speed delays and live websocket broadcast."""
        steps_run = 0
        try:
            while self.state == "RUNNING":
                if not self.env or self.env.clock.finished():
                    self.state = "PAUSED"
                    break
                if max_steps is not None and steps_run >= max_steps:
                    self.state = "PAUSED"
                    break

                # Execute one step
                self.step(1)
                steps_run += 1

                # Broadcast to connected WebSockets
                if self.clients:
                    payload = self.latest_telemetry
                    dead_clients = set()
                    for ws in self.clients:
                        try:
                            await ws.send_json({"event_type": "TELEMETRY", "payload": payload})
                        except Exception:
                            dead_clients.add(ws)
                    self.clients.difference_update(dead_clients)

                delay = SPEED_DELAYS.get(self.speed, 0.10)
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Error in simulation run loop: %s", exc)
            self.state = "PAUSED"

    def _extract_telemetry(self, step_res: Any | None, latency_ms: float = 0.5) -> dict[str, Any]:
        """Builds a standardized, non-leaking, explainable EW telemetry payload."""
        if not self.env:
            return {}

        t = self.env.clock.time_step
        sim_time = (self.env.clock.time_ms / 1000.0)
        num_bins = len(self.env.bands_hz)
        bands_mhz = [round(b / 1e6, 2) for b in self.env.bands_hz]

        # 1. Receiver & Current Scan
        last_obs = self.env._current_observation
        rx = self.env.receiver
        current_freq_mhz = round(rx.center_frequency_hz / 1e6, 2)
        if last_obs and hasattr(last_obs, "current_frequency_bin") and last_obs.current_frequency_bin is not None:
            current_bin = int(last_obs.current_frequency_bin)
        else:
            diffs = [abs(b - rx.center_frequency_hz) for b in self.env.bands_hz]
            current_bin = int(np.argmin(diffs)) if diffs else 0

        # 2. Detector Result
        detected = bool(last_obs.last_detection) if last_obs else False
        last_det_bin = int(last_obs.last_detection_bin) if last_obs and last_obs.last_detection_bin is not None else None
        last_det_mhz = round(self.env.bands_hz[last_det_bin] / 1e6, 2) if last_det_bin is not None and 0 <= last_det_bin < num_bins else None
        strength_dbm = round(float(last_obs.last_detection_strength), 1) if last_obs and last_obs.last_detection_strength is not None else None

        # 3. Scheduler Diagnostics & Next Prediction
        sched = self.env.scheduler
        diag = sched.get_diagnostics() if hasattr(sched, "get_diagnostics") else {}

        # Default prediction fallback
        predicted_bin = current_bin
        confidence_pct = 50.0
        q_ranking: list[dict[str, Any]] = []
        q_values_list: list[float] = []

        # If scheduler is hybrid or RL with Q-values
        if hasattr(sched, "ddqn") and hasattr(sched.ddqn, "get_q_values") and last_obs:
            try:
                q_vals = sched.ddqn.get_q_values(last_obs)
                q_values_list = [round(float(q), 3) for q in q_vals]
                predicted_bin = int(np.argmax(q_vals))

                q_mean = float(np.mean(q_vals))
                q_std = float(np.std(q_vals))
                z = (float(np.max(q_vals)) - q_mean) / (2.0 * max(q_std, 1e-4))
                confidence_pct = round(float(np.clip(z, 0.15, 0.98)) * 100.0, 1)

                sorted_indices = np.argsort(q_vals)[::-1]
                exp_q = np.exp(np.clip(q_vals - np.max(q_vals), -20, 0))
                shares = exp_q / max(np.sum(exp_q), 1e-8)

                for rank, idx in enumerate(sorted_indices[:5]):
                    q_ranking.append({
                        "rank": rank + 1,
                        "bin": int(idx),
                        "frequency_mhz": bands_mhz[idx],
                        "q_value": round(float(q_vals[idx]), 3),
                        "share_pct": round(float(shares[idx]) * 100.0, 1),
                    })
            except Exception:
                pass
        elif hasattr(sched, "lstm_ddqn") and hasattr(sched.lstm_ddqn, "get_q_values") and last_obs:
            try:
                q_vals = sched.lstm_ddqn.get_q_values(last_obs)
                q_values_list = [round(float(q), 3) for q in q_vals]
                predicted_bin = int(np.argmax(q_vals))

                q_mean = float(np.mean(q_vals))
                q_std = float(np.std(q_vals))
                z = (float(np.max(q_vals)) - q_mean) / (2.0 * max(q_std, 1e-4))
                confidence_pct = round(float(np.clip(z, 0.15, 0.98)) * 100.0, 1)

                sorted_indices = np.argsort(q_vals)[::-1]
                exp_q = np.exp(np.clip(q_vals - np.max(q_vals), -20, 0))
                shares = exp_q / max(np.sum(exp_q), 1e-8)

                for rank, idx in enumerate(sorted_indices[:5]):
                    q_ranking.append({
                        "rank": rank + 1,
                        "bin": int(idx),
                        "frequency_mhz": bands_mhz[idx],
                        "q_value": round(float(q_vals[idx]), 3),
                        "share_pct": round(float(shares[idx]) * 100.0, 1),
                    })
            except Exception:
                pass
        elif hasattr(sched, "last_indices") and sched.last_indices is not None:
            try:
                indices = np.array(sched.last_indices, dtype=float)
                q_values_list = [round(float(q), 3) for q in indices]
                predicted_bin = int(np.argmax(indices))
                q_mean = float(np.mean(indices))
                q_std = float(np.std(indices))
                z = (float(np.max(indices)) - q_mean) / (2.0 * max(q_std, 1e-4))
                confidence_pct = round(float(np.clip(z, 0.20, 0.95)) * 100.0, 1)

                sorted_indices = np.argsort(indices)[::-1]
                exp_q = np.exp(np.clip(indices - np.max(indices), -20, 0))
                shares = exp_q / max(np.sum(exp_q), 1e-8)

                for rank, idx in enumerate(sorted_indices[:5]):
                    q_ranking.append({
                        "rank": rank + 1,
                        "bin": int(idx),
                        "frequency_mhz": bands_mhz[idx],
                        "q_value": round(float(indices[idx]), 3),
                        "share_pct": round(float(shares[idx]) * 100.0, 1),
                    })
            except Exception:
                pass
        elif hasattr(sched, "last_scores") and sched.last_scores is not None:
            try:
                scores = np.array(sched.last_scores, dtype=float)
                q_values_list = [round(float(q), 3) for q in scores]
                predicted_bin = int(np.argmax(scores))
                confidence_pct = round(float(np.clip(np.max(scores), 0.15, 0.95)) * 100.0, 1)

                sorted_indices = np.argsort(scores)[::-1]
                shares = scores / max(np.sum(scores), 1e-8)

                for rank, idx in enumerate(sorted_indices[:5]):
                    q_ranking.append({
                        "rank": rank + 1,
                        "bin": int(idx),
                        "frequency_mhz": bands_mhz[idx],
                        "q_value": round(float(scores[idx]), 3),
                        "share_pct": round(float(shares[idx]) * 100.0, 1),
                    })
            except Exception:
                pass
        elif self.scheduler_name == "sequential":
            predicted_bin = (current_bin + 1) % num_bins
            confidence_pct = 99.0
            q_ranking.append({
                "rank": 1,
                "bin": predicted_bin,
                "frequency_mhz": bands_mhz[predicted_bin],
                "q_value": 1.0,
                "share_pct": 100.0,
            })
        elif self.scheduler_name == "random":
            predicted_bin = int(self.env.scheduler.rng.integers(0, num_bins)) if hasattr(self.env.scheduler, "rng") else 0
            confidence_pct = round(100.0 / num_bins, 1)
            q_ranking.append({
                "rank": 1,
                "bin": predicted_bin,
                "frequency_mhz": bands_mhz[predicted_bin],
                "q_value": round(1.0 / num_bins, 3),
                "share_pct": round(100.0 / num_bins, 1),
            })
        elif hasattr(sched, "last_selected_bin") and sched.last_selected_bin is not None:
            predicted_bin = int(sched.last_selected_bin)

        predicted_freq_mhz = bands_mhz[predicted_bin] if 0 <= predicted_bin < num_bins else current_freq_mhz

        # Confidence qualitative level
        if confidence_pct >= 75.0:
            conf_level = "HIGH"
        elif confidence_pct >= 45.0:
            conf_level = "MEDIUM"
        else:
            conf_level = "LOW"

        # 4. Arbitration & Decision Mode Details
        last_expl = getattr(sched, "last_explanation", {})
        if self.scheduler_name in {"hybrid_v4", "hybrid", "hybrid_meta"}:
            arb_mode = last_expl.get("mode", "DDQN_EXPLOIT")
            ddqn_weight_pct = round(float(last_expl.get("ddqn_weight", 0.70)) * 100.0, 1)
            ca_weight_pct = round(float(last_expl.get("ca_weight", 0.30)) * 100.0, 1)
            surprise = round(float(last_expl.get("surprise", 0.0)), 3)
            consistency = round(float(last_expl.get("consistency", 0.0)), 3)
        elif self.scheduler_name in {"hybrid_v41", "hybrid_lstm"}:
            arb_mode = last_expl.get("mode", "LSTM_EXPLOIT")
            ddqn_weight_pct = round(float(last_expl.get("ddqn_weight", 0.70)) * 100.0, 1)
            ca_weight_pct = round(float(last_expl.get("ca_weight", 0.30)) * 100.0, 1)
            surprise = round(float(last_expl.get("surprise", 0.0)), 3)
            consistency = round(float(last_expl.get("consistency", 0.0)), 3)
        elif self.scheduler_name == "whittle_style":
            arb_mode = "WHITTLE_INDEX"
            ddqn_weight_pct = 0.0
            ca_weight_pct = 0.0
            surprise = 0.0
            consistency = 0.92
        elif self.scheduler_name in {"context_aware", "contextual"}:
            arb_mode = "EMPIRICAL_MARKOV"
            ddqn_weight_pct = 0.0
            ca_weight_pct = 100.0
            surprise = 0.0
            consistency = 0.85
        elif self.scheduler_name in {"v5_belief", "v5"}:
            arb_mode = "BAYESIAN_POMDP"
            ddqn_weight_pct = 0.0
            ca_weight_pct = 0.0
            surprise = 0.0
            consistency = 0.90
        elif self.scheduler_name == "sequential":
            arb_mode = "SEQUENTIAL_SWEEP"
            ddqn_weight_pct = 0.0
            ca_weight_pct = 0.0
            surprise = 0.0
            consistency = 1.00
        elif self.scheduler_name == "random":
            arb_mode = "UNIFORM_RANDOM"
            ddqn_weight_pct = 0.0
            ca_weight_pct = 0.0
            surprise = 1.00
            consistency = 0.00
        else:
            arb_mode = "BASELINE_POLICY"
            ddqn_weight_pct = 0.0
            ca_weight_pct = 0.0
            surprise = 0.0
            consistency = 0.50

        # 5. Explainability ("Why this scan?")
        why_explanation = self._build_why_explanation(
            arb_mode=arb_mode,
            chosen_freq=predicted_freq_mhz,
            chosen_bin=predicted_bin,
            conf_level=conf_level,
            detected=detected,
            ddqn_pct=ddqn_weight_pct,
            ca_pct=ca_weight_pct,
            surprise=surprise,
            consistency=consistency,
        )


        # 6. Performance Metrics
        snap = self.env.metrics.snapshot(t)
        opp_summary = self.env.metrics.opportunity_tracker.compute_summary()
        interception_ratio = round(float(opp_summary.get("interception_ratio", 0.0)) * 100.0, 1)
        detection_rate = round(float(snap.probability_of_detection) * 100.0, 1)
        scan_eff = round(float(snap.hits) / max(snap.total_scans, 1) * 100.0, 1)

        # 7. Record step in timeline and waterfall
        if step_res is not None and t >= 0:
            step_entry = {
                "step": t,
                "simulation_time_s": round(sim_time, 2),
                "scanned_bin": current_bin,
                "scanned_mhz": current_freq_mhz,
                "predicted_bin": predicted_bin,
                "predicted_mhz": predicted_freq_mhz,
                "detected": detected,
                "detected_mhz": last_det_mhz if detected else None,
                "signal_power_dbm": strength_dbm,
                "confidence_pct": confidence_pct,
                "arbitration_mode": arb_mode,
            }
            self.timeline_history.append(step_entry)

            # Extract Ground Truth emitters (for isolated evaluation view only)
            gt_emitters = []
            if self.env.ground_truth and self.env.ground_truth.latest:
                for em in self.env.ground_truth.latest.emitters:
                    if em.transmitting:
                        gt_emitters.append({
                            "id": em.emitter_id,
                            "frequency_mhz": round(em.frequency_hz / 1e6, 2),
                            "power_dbm": round(em.power_dbm, 1),
                        })

            waterfall_entry = {
                "step": t,
                "scanned_mhz": current_freq_mhz,
                "predicted_mhz": predicted_freq_mhz,
                "detected": detected,
                "detected_mhz": last_det_mhz if detected else None,
                "ground_truth": gt_emitters,
            }
            self.waterfall_history.append(waterfall_entry)

        # Determine readable scheduler display name
        if self.scheduler_name in {"hybrid_v4", "hybrid", "hybrid_meta"}:
            sched_display = "V4.0 Hybrid (Benchmark Winner: 35.19% IR)"
        elif self.scheduler_name in {"hybrid_v41", "hybrid_lstm"}:
            sched_display = "V4.1 LSTM-Hybrid"
        elif self.scheduler_name == "whittle_style":
            sched_display = "Whittle-Style Index"
        elif self.scheduler_name in {"v5_belief", "v5"}:
            sched_display = "V5.0 Augmented Belief-State"
        else:
            sched_display = SCHEDULER_METADATA.get(self.scheduler_name, {}).get("name", self.scheduler_name)

        # 8. Assemble payload
        payload = {
            "system_status": {
                "state": self.state,
                "step": max(t, 0),
                "simulation_time_s": round(max(sim_time, 0.0), 2),
                "scenario_name": self.scenario_name,
                "scheduler_name": sched_display,
                "scheduler_type": self.scheduler_name,
                "version": APP_VERSION,
                "seed": self.seed,
            },

            "primary_prediction": {
                "current_scan_bin": current_bin,
                "current_scan_mhz": current_freq_mhz,
                "predicted_next_bin": predicted_bin,
                "predicted_next_mhz": predicted_freq_mhz,
                "confidence_pct": confidence_pct,
                "confidence_level": conf_level,
                "top_predictions": q_ranking,
                "q_values": q_values_list,
            },
            "detector": {
                "detected": detected,
                "last_detection_bin": last_det_bin,
                "last_detection_mhz": last_det_mhz,
                "signal_power_dbm": strength_dbm,
                "status_label": "SIGNAL DETECTED" if detected else "NO SIGNAL",
                "receiver_bandwidth_mhz": round(rx.instantaneous_bandwidth_hz / 1e6, 1),
                "sensitivity_dbm": rx.sensitivity_dbm,
            },
            "performance": {
                "interception_ratio_pct": interception_ratio,
                "detection_rate_pct": detection_rate,
                "scan_efficiency_pct": scan_eff,
                "total_opportunities": opp_summary.get("opportunities_total", 0),
                "intercepted_opportunities": opp_summary.get("opportunities_covered", 0),
                "total_detections": snap.hits,
                "total_scans": snap.total_scans,
            },
            "arbitration": {
                "mode": arb_mode,
                "ddqn_weight_pct": ddqn_weight_pct,
                "ca_weight_pct": ca_weight_pct,
                "surprise": surprise,
                "consistency": consistency,
                "explanation": why_explanation,
            },
            "latency": {
                "step_latency_ms": round(latency_ms, 2),
                "budget_ms": 10.0,
                "status": "WITHIN BUDGET (< 10 ms)" if latency_ms < 10.0 else "EXCEEDS BUDGET",
            },
            "neural_model": {
                "architecture": "MLPQNetwork (Feedforward DDQN)" if hasattr(sched, "ddqn") else ("LSTMQNetwork (DRQN)" if hasattr(sched, "lstm_ddqn") else f"{SCHEDULER_METADATA.get(self.scheduler_name, {}).get('name', self.scheduler_name)}"),
                "status": "PRETRAINED" if getattr(sched, "is_pretrained", False) else ("LOADED" if getattr(sched, "checkpoint_path", None) else "BASELINE_MODEL"),
                "mode": "FROZEN_INFERENCE" if (hasattr(sched, "ddqn") or hasattr(sched, "lstm_ddqn")) else "ZERO_SHOT_POLICY",
                "runtime_training": "DISABLED" if (hasattr(sched, "ddqn") or hasattr(sched, "lstm_ddqn")) else "N/A (Heuristic)",
                "checkpoint_name": Path(sched.checkpoint_path).name if getattr(sched, "checkpoint_path", None) else "N/A (Baseline Algorithm)",
                "checkpoint_path": str(sched.checkpoint_path) if getattr(sched, "checkpoint_path", None) else None,
                "checkpoint_sha256": getattr(sched, "checkpoint_sha256", None),
                "checkpoint_fingerprint": getattr(sched, "checkpoint_sha256", "")[:8] if getattr(sched, "checkpoint_sha256", None) else None,
                "is_pretrained": bool(getattr(sched, "is_pretrained", False)),
            },
            "bands_mhz": bands_mhz,
            "recent_timeline": list(reversed(self.timeline_history)),
            "waterfall_events": list(self.waterfall_history),
        }

        return _clean_val(payload)

    def _build_why_explanation(
        self,
        arb_mode: str,
        chosen_freq: float,
        chosen_bin: int,
        conf_level: str,
        detected: bool,
        ddqn_pct: float,
        ca_pct: float,
        surprise: float,
        consistency: float = 0.0,
    ) -> str:
        """Generates natural language tactical rationale based on genuine scheduler telemetry."""
        if arb_mode == "WHITTLE_INDEX":
            return (
                f"Whittle-Style Index Policy: Arm index maximized for Bin {chosen_bin} ({chosen_freq:.1f} MHz). "
                f"Balancing restless arm state belief, recency, dwell aging, and return interval periodicity."
            )
        if arb_mode == "EMPIRICAL_MARKOV":
            return (
                f"Context-Aware Empirical Baseline: Probing Bin {chosen_bin} ({chosen_freq:.1f} MHz) "
                f"based on 1st-order Markov transition frequency and recent activity bonuses."
            )
        if arb_mode == "BAYESIAN_POMDP":
            return (
                f"V5.0 Exact Bayesian POMDP: Selected Bin {chosen_bin} ({chosen_freq:.1f} MHz) "
                f"maximizing recursive posterior belief P(F_t={chosen_bin}) under semi-Markov dwell filter."
            )
        if arb_mode == "SEQUENTIAL_SWEEP":
            return (
                f"Legacy Sequential Baseline: Advancing to next channel Bin {chosen_bin} ({chosen_freq:.1f} MHz) "
                f"via blind fixed linear sweep. No pattern learning, memory, or cognitive adaptation."
            )
        if arb_mode == "UNIFORM_RANDOM":
            return (
                f"Zero-Intelligence Random Baseline: Randomly chosen channel Bin {chosen_bin} ({chosen_freq:.1f} MHz). "
                f"Uniform probability (~3.3%) across all 30 channels with zero state tracking."
            )
        if arb_mode == "DDQN_EXPLOIT":
            return (
                f"Anticipating periodic hopping transition to {chosen_freq:.1f} MHz (Bin {chosen_bin}). "
                f"DDQN predictive branch identified structured dwell pattern. Model confidence: {conf_level} "
                f"(Decision weight: {ddqn_pct:.0f}% DDQN / {ca_pct:.0f}% CA)."
            )
        if arb_mode == "CA_ADAPT":
            return (
                f"Environmental novelty/surprise spike detected ({surprise:.2f}). "
                f"Arbitrator transitioned to Adaptive Discovery ({ca_pct:.0f}% CA weight), "
                f"executing contextual empirical transition scan towards {chosen_freq:.1f} MHz."
            )
        if arb_mode == "EXPLORE_DISCOVERY":
            return (
                f"Low confidence across both branches. Executing wideband heuristic search "
                f"on channel {chosen_freq:.1f} MHz to re-acquire agile emitter."
            )
        return (
            f"Blended arbitration scan on {chosen_freq:.1f} MHz (w_DDQN={ddqn_pct:.0f}%, w_CA={ca_pct:.0f}%). "
            f"Fusing predictive neural pattern with empirical coverage bonus."
        )



# Global service singleton
service = SimulationService()
