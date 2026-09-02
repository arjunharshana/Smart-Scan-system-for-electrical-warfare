from __future__ import annotations

import os
from typing import Any
import pandas as pd
import yaml

from rf_environment.environment.scenario import load_scenario


def get_scenarios_directory() -> str:
    """Returns the absolute path to the scenarios directory."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, "rf_environment", "scenarios")


def list_available_scenarios() -> list[str]:
    """Returns a sorted list of scenario YAML filenames."""
    s_dir = get_scenarios_directory()
    if not os.path.exists(s_dir):
        return []
    return sorted([f for f in os.listdir(s_dir) if f.endswith(".yaml") or f.endswith(".yml")])


def load_scenario_file(filename: str) -> dict[str, Any]:
    """Loads a scenario by filename or full path."""
    if os.path.isabs(filename) or os.path.exists(filename):
        return load_scenario(filename)
    full_path = os.path.join(get_scenarios_directory(), filename)
    return load_scenario(full_path)


def format_frequency(hz: float) -> str:
    """Formats frequency in Hz to readable MHz/GHz string."""
    if hz >= 1e9:
        return f"{hz / 1e9:.3f} GHz"
    if hz >= 1e6:
        return f"{hz / 1e6:.2f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.1f} kHz"
    return f"{hz:.0f} Hz"


def format_bandwidth(hz: float) -> str:
    """Formats bandwidth in Hz."""
    if hz >= 1e6:
        return f"{hz / 1e6:.1f} MHz"
    if hz >= 1e3:
        return f"{hz / 1e3:.1f} kHz"
    return f"{hz:.0f} Hz"


def build_time_series_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Extracts step-by-step metric trends into a Pandas DataFrame."""
    records = []
    cumulative_reward = 0.0
    rolling_window = 20
    hits_window = []

    for r in results:
        t = r["timestamp"]
        step_reward = r["outcome"].get("reward", 0.0)
        cumulative_reward += step_reward
        outcome = r["outcome"].get("outcome", "UNKNOWN")
        is_hit = 1 if outcome == "HIT" else 0
        hits_window.append(is_hit)
        if len(hits_window) > rolling_window:
            hits_window.pop(0)

        rolling_hit_rate = sum(hits_window) / len(hits_window) if hits_window else 0.0
        metrics = r.get("metrics", {})

        records.append({
            "Time Step": t,
            "Outcome": outcome,
            "Step Reward": step_reward,
            "Cumulative Reward": cumulative_reward,
            "Rolling Hit Rate": rolling_hit_rate,
            "Hits": metrics.get("hits", 0),
            "Misses": metrics.get("misses", 0),
            "False Alarms": metrics.get("false_alarms", 0),
            "Correct Rejections": metrics.get("correct_rejections", 0),
            "P(Detection)": metrics.get("probability_of_detection", 0.0),
            "P(False Alarm)": metrics.get("probability_of_false_alarm", 0.0),
            "Interception Ratio": metrics.get("interception_ratio", 0.0),
            "Average Reward": metrics.get("average_reward", 0.0),
            "Receiver Freq (MHz)": r["receiver"].get("center_frequency_hz", 0.0) / 1e6,
        })

    return pd.DataFrame(records)


def build_waterfall_dataframe(waterfall_items: list[dict[str, Any]]) -> pd.DataFrame:
    """Builds a DataFrame for waterfall visualization with emitter states & scans."""
    records = []
    for item in waterfall_items:
        t = item["timestamp"]
        # Emitters
        for gt in item.get("ground_truth", []):
            if gt.get("transmitting", False):
                records.append({
                    "Time Step": t,
                    "Frequency (MHz)": gt["frequency_hz"] / 1e6,
                    "Power (dBm)": gt.get("power_dbm", -30),
                    "Emitter ID": gt["emitter_id"],
                    "Category": f"Emitter {gt['emitter_id']} (Tx)",
                    "Marker Type": "Emitter",
                    "Status": "Transmitting",
                })
        
        # Receiver scan
        rx_freq = item.get("receiver_frequency_hz", 0.0)
        detected = item.get("detected", False)
        status_label = "🎯 Receiver Scan (Hit)" if detected else "❌ Receiver Scan (Miss)"
        records.append({
            "Time Step": t,
            "Frequency (MHz)": rx_freq / 1e6,
            "Power (dBm)": 0,
            "Emitter ID": "Receiver",
            "Category": status_label,
            "Marker Type": "Receiver",
            "Status": "Hit" if detected else "Miss",
        })

    return pd.DataFrame(records)


def build_arm_stats_dataframe(scheduler_state: dict[str, Any]) -> pd.DataFrame:
    """Extracts Multi-Armed Bandit / Scheduler arm statistics."""
    arm_stats = scheduler_state.get("arm_stats", [])
    if not arm_stats:
        return pd.DataFrame()
    
    rows = []
    for arm in arm_stats:
        freq = arm.get("frequency_hz", 0.0)
        pulls = arm.get("pulls", arm.get("visits", arm.get("count", 0)))
        reward_sum = arm.get("total_reward", arm.get("reward", 0.0))
        mean_reward = arm.get("mean_reward", (reward_sum / pulls) if pulls > 0 else 0.0)
        alpha = arm.get("alpha")
        beta = arm.get("beta")

        row = {
            "Frequency (MHz)": f"{freq / 1e6:.1f}",
            "Center Freq (Hz)": freq,
            "Scans / Pulls": pulls,
            "Total Reward": reward_sum,
            "Mean Reward": mean_reward,
        }
        if alpha is not None and beta is not None:
            row["Beta Alpha (Hits)"] = alpha
            row["Beta Beta (Misses)"] = beta
            row["Estimated Prob"] = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0
        
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
