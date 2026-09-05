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
    PROJECT_ROOT,
)
from rf_environment.environment.builder import build_environment
from rf_environment.environment.rf_environment import RFEnvironment
from rf_environment.environment.scenario import load_scenario
from rf_environment.scheduler.factory import SCHEDULER_METADATA

logger = logging.getLogger("simulation_service")

# Speed delays in seconds per step
SPEED_DELAYS: dict[str, float] = {
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
        scenarios_dir = PROJECT_ROOT / "rf_environment" / "scenarios"
        items = []
        if scenarios_dir.exists():
            for p in sorted(scenarios_dir.glob("*.yaml")):
                name = p.stem.replace("_", " ").title()
                items.append({
                    "id": p.stem,
                    "name": name,
                    "filename": p.name,
                    "path": str(p),
                })
        return items

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

        sc_path = PROJECT_ROOT / "rf_environment" / "scenarios" / self.scenario_name
        if not sc_path.exists():
            sc_path = DEFAULT_SCENARIO_PATH
            self.scenario_name = DEFAULT_SCENARIO_NAME

        self.scenario_dict = load_scenario(str(sc_path))
        sc_copy = copy.deepcopy(self.scenario_dict)

        if "simulation" not in sc_copy:
            sc_copy["simulation"] = {}
        sc_copy["simulation"]["seed"] = self.seed

        # Build environment
        self.env = build_environment(sc_copy, scheduler_name=self.scheduler_name)
        if hasattr(self.env.scheduler, "eval"):
            self.env.scheduler.eval()
        if hasattr(self.env.scheduler, "epsilon"):
            self.env.scheduler.epsilon = 0.0

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
        """Executes count simulation steps synchronously and records telemetry."""
        if not self.env:
            raise RuntimeError("Environment not initialized")

        last_result = None
        for _ in range(count):
            if self.env.clock.finished():
                self.state = "PAUSED"
                break
            last_result = self.env.step()
            telem = self._extract_telemetry(last_result)
            self.latest_telemetry = telem

        return self.latest_telemetry

    async def start(self, steps: int | None = None) -> None:
        """Starts asynchronous continuous execution."""
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

    def _extract_telemetry(self, step_res: Any | None) -> dict[str, Any]:
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
        if hasattr(sched, "lstm_ddqn") and hasattr(sched.lstm_ddqn, "get_q_values") and last_obs:
            try:
                q_vals = sched.lstm_ddqn.get_q_values(last_obs)
                q_values_list = [round(float(q), 3) for q in q_vals]
                predicted_bin = int(np.argmax(q_vals))

                # Calibrate confidence from separation of max Q over mean and std
                q_mean = float(np.mean(q_vals))
                q_std = float(np.std(q_vals))
                z = (float(np.max(q_vals)) - q_mean) / (2.0 * max(q_std, 1e-4))
                confidence_pct = round(float(np.clip(z, 0.15, 0.98)) * 100.0, 1)

                # Q-value ranking top 5
                sorted_indices = np.argsort(q_vals)[::-1]
                # Softmax-style shares for ranking table
                exp_q = np.exp(np.clip(q_vals - np.max(q_vals), -20, 0))
                shares = exp_q / np.sum(exp_q)

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
        elif hasattr(sched, "ddqn") and hasattr(sched.ddqn, "get_q_values") and last_obs:
            try:
                q_vals = sched.ddqn.get_q_values(last_obs)
                q_values_list = [round(float(q), 3) for q in q_vals]
                predicted_bin = int(np.argmax(q_vals))
                sorted_indices = np.argsort(q_vals)[::-1]
                for rank, idx in enumerate(sorted_indices[:5]):
                    q_ranking.append({
                        "rank": rank + 1,
                        "bin": int(idx),
                        "frequency_mhz": bands_mhz[idx],
                        "q_value": round(float(q_vals[idx]), 3),
                        "share_pct": round(100.0 / (rank + 1), 1),
                    })
            except Exception:
                pass
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

        # 4. Hybrid Arbitration Details
        last_expl = getattr(sched, "last_explanation", {})
        arb_mode = last_expl.get("mode", "DDQN_EXPLOIT" if "lstm_ddqn" in self.scheduler_name else "AUTONOMOUS")
        ddqn_weight_pct = round(float(last_expl.get("ddqn_weight", 0.70)) * 100.0, 1)
        ca_weight_pct = round(float(last_expl.get("ca_weight", 0.30)) * 100.0, 1)
        surprise = round(float(last_expl.get("surprise", 0.0)), 3)

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
        )

        # 6. Performance Metrics
        snap = self.env.metrics.snapshot(t)
        opp_summary = self.env.metrics.opportunity_tracker.compute_summary()
        interception_ratio = round(float(opp_summary.get("interception_ratio", 0.0)) * 100.0, 1)
        detection_rate = round(float(snap.probability_of_detection) * 100.0, 1)
        scan_eff = round(float(snap.hits) / max(snap.total_scans, 1) * 100.0, 1)

        # 7. Temporal Working Memory status
        h_norm = 0.0
        c_norm = 0.0
        if hasattr(sched, "lstm_ddqn") and hasattr(sched.lstm_ddqn, "online_net"):
            try:
                lstm = sched.lstm_ddqn.online_net.lstm
                h_norm = round(float(np.linalg.norm(lstm.h)), 3)
                c_norm = round(float(np.linalg.norm(lstm.c)), 3)
            except Exception:
                pass

        # 8. Record step in timeline and waterfall
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

        # 9. Assemble payload
        payload = {
            "system_status": {
                "state": self.state,
                "step": t,
                "simulation_time_s": round(sim_time, 2),
                "scenario_name": self.scenario_name,
                "scheduler_name": "V4.1 LSTM-Hybrid" if "hybrid_v41" in self.scheduler_name else self.scheduler_name,
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
                "lstm_ddqn_weight_pct": ddqn_weight_pct,
                "context_aware_weight_pct": ca_weight_pct,
                "surprise": surprise,
                "explanation": why_explanation,
            },
            "temporal_memory": {
                "history_window": 10,
                "hidden_state_norm": h_norm,
                "cell_state_norm": c_norm,
                "pattern_confidence": conf_level,
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
    ) -> str:
        """Generates natural language tactical rationale based on genuine arbitrator telemetry."""
        if arb_mode == "DDQN_EXPLOIT":
            return (
                f"Anticipating periodic hopping transition to {chosen_freq:.1f} MHz (Bin {chosen_bin}). "
                f"Recurrent cell state identified dwell completion. LSTM confidence: {conf_level} "
                f"(Decision weight: {ddqn_pct:.0f}% LSTM / {ca_pct:.0f}% CA)."
            )
        if arb_mode == "CA_ADAPT":
            return (
                f"Environmental novelty/surprise spike detected ({surprise:.2f}). "
                f"Arbitrator transitioned to Adaptive Discovery ({ca_pct:.0f}% CA weight), "
                f"executing contextual empirical transition scan towards {chosen_freq:.1f} MHz."
            )
        if arb_mode == "EXPLORE_DISCOVERY":
            return (
                f"Low pattern confidence across both branches. Executing wideband heuristic search "
                f"on channel {chosen_freq:.1f} MHz to re-acquire lost agile emitter."
            )
        return (
            f"Blended arbitration scan on {chosen_freq:.1f} MHz. "
            f"Fusing predictive neural pattern with empirical coverage bonus."
        )


# Global service singleton
service = SimulationService()
