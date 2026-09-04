from __future__ import annotations

import os
from typing import Any
import altair as alt
import numpy as np
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
    rewards_window = []

    for r in results:
        t = r["timestamp"]
        step_reward = r["outcome"].get("reward", 0.0)
        cumulative_reward += step_reward
        outcome = r["outcome"].get("outcome", "UNKNOWN")
        is_hit = 1 if outcome == "HIT" else 0
        hits_window.append(is_hit)
        rewards_window.append(step_reward)
        if len(hits_window) > rolling_window:
            hits_window.pop(0)
            rewards_window.pop(0)

        rolling_hit_rate = sum(hits_window) / len(hits_window) if hits_window else 0.0
        rolling_reward = sum(rewards_window) / len(rewards_window) if rewards_window else 0.0
        metrics = r.get("metrics", {})

        records.append({
            "Time Step": t,
            "Outcome": outcome,
            "Step Reward": step_reward,
            "Cumulative Reward": cumulative_reward,
            "Rolling Interception Rate": rolling_hit_rate,
            "Rolling Reward": rolling_reward,
            "Hits": metrics.get("hits", 0),
            "Misses": metrics.get("misses", 0),
            "False Alarms": metrics.get("false_alarms", 0),
            "Correct Rejections": metrics.get("correct_rejections", 0),
            "P(Detection)": metrics.get("probability_of_detection", 0.0),
            "P(False Alarm)": metrics.get("probability_of_false_alarm", 0.0),
            "Interception Ratio": metrics.get("interception_ratio", 0.0),
            "Opportunity Coverage": metrics.get("opportunity_coverage", 0.0),
            "Detection Given Coverage": metrics.get("detection_given_coverage", 0.0),
            "Average Intercept Latency": metrics.get("average_intercept_time"),
            "Receiver Freq (MHz)": r["receiver"].get("center_frequency_hz", 0.0) / 1e6,
        })

    return pd.DataFrame(records)


def build_waterfall_dataframe(waterfall_items: list[dict[str, Any]]) -> pd.DataFrame:
    """Builds structured records for waterfall visualization."""
    records = []
    for item in waterfall_items:
        t = item["timestamp"]
        rx_freq = item.get("receiver_frequency_hz", 0.0) / 1e6
        rx_bw = item.get("receiver_bandwidth_hz", 20e6) / 1e6
        rx_min = rx_freq - rx_bw / 2.0
        rx_max = rx_freq + rx_bw / 2.0
        detected = item.get("detected", False)
        outcome = item.get("outcome", "HIT" if detected else "MISS")
        reason = item.get("diagnostic_reason", "")

        # Emitter transmissions at this step
        for gt in item.get("ground_truth", []):
            if gt.get("transmitting", False):
                records.append({
                    "Time Step": t,
                    "Frequency (MHz)": gt["frequency_hz"] / 1e6,
                    "Bandwidth (MHz)": gt.get("bandwidth_hz", 2e6) / 1e6,
                    "Power (dBm)": gt.get("power_dbm", -20.0),
                    "Category": f"Emitter {gt['emitter_id']} (Tx)",
                    "Layer": "Emitter",
                    "Status": "Transmitting",
                    "Outcome": outcome,
                    "Receiver Freq (MHz)": rx_freq,
                    "Receiver Freq Min": rx_min,
                    "Receiver Freq Max": rx_max,
                    "Diagnostic": reason,
                })

        # Receiver scan record
        status_label = "🎯 Scan Hit" if outcome == "HIT" else (
            "⚠️ False Alarm" if outcome == "FALSE_ALARM" else (
                "❌ Scan Miss" if outcome == "MISS" else "🔍 Off-Frequency Scan"
            )
        )
        records.append({
            "Time Step": t,
            "Frequency (MHz)": rx_freq,
            "Bandwidth (MHz)": rx_bw,
            "Power (dBm)": 0.0,
            "Category": status_label,
            "Layer": "Receiver",
            "Status": outcome,
            "Outcome": outcome,
            "Receiver Freq (MHz)": rx_freq,
            "Receiver Freq Min": rx_min,
            "Receiver Freq Max": rx_max,
            "Diagnostic": reason,
        })

    return pd.DataFrame(records)


def build_waterfall_altair_chart(waterfall_df: pd.DataFrame, height: int = 420) -> alt.LayerChart:
    """Builds an interactive Altair layered Time x Frequency Waterfall chart.
    
    Layers:
    1. Receiver scan window (shaded area [rx_min, rx_max] over time)
    2. Emitter active transmission trajectories (colored points/lines)
    3. Detection outcomes markers (Hits 🎯, Misses ❌, False Alarms ⚠️)
    """
    if waterfall_df.empty:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_point()

    # Split into receiver scan window and emitter events
    rx_df = waterfall_df[waterfall_df["Layer"] == "Receiver"].drop_duplicates(subset=["Time Step"])
    emitter_df = waterfall_df[waterfall_df["Layer"] == "Emitter"]

    # Base X-axis time scale
    x_enc = alt.X("Time Step:Q", title="Time Step (Steps)")
    y_enc = alt.Y("Frequency (MHz):Q", title="Frequency (MHz)", scale=alt.Scale(zero=False))

    # 1. Receiver scan band (shaded bar [rx_min, rx_max])
    rx_bands = (
        alt.Chart(rx_df)
        .mark_rule(opacity=0.35, strokeWidth=12, color="#48cae4")
        .encode(
            x=alt.X("Time Step:Q"),
            y=alt.Y("Receiver Freq Min:Q"),
            y2=alt.Y2("Receiver Freq Max:Q"),
            tooltip=[
                alt.Tooltip("Time Step:Q"),
                alt.Tooltip("Receiver Freq (MHz):Q", format=".1f"),
                alt.Tooltip("Outcome:N"),
                alt.Tooltip("Diagnostic:N"),
            ],
        )
    )

    # 2. Emitter active signal points
    emitter_points = (
        alt.Chart(emitter_df)
        .mark_circle(size=70, opacity=0.9)
        .encode(
            x=x_enc,
            y=y_enc,
            color=alt.Color("Category:N", scale=alt.Scale(scheme="category10"), legend=alt.Legend(title="Emitter")),
            tooltip=[
                alt.Tooltip("Time Step:Q"),
                alt.Tooltip("Category:N"),
                alt.Tooltip("Frequency (MHz):Q", format=".1f"),
                alt.Tooltip("Power (dBm):Q", format=".1f"),
            ],
        )
    )

    # 3. Detection outcomes overlay
    rx_outcomes = (
        alt.Chart(rx_df)
        .mark_point(filled=True, size=65)
        .encode(
            x=x_enc,
            y=alt.Y("Receiver Freq (MHz):Q"),
            color=alt.Color(
                "Category:N",
                scale=alt.Scale(
                    domain=["🎯 Scan Hit", "❌ Scan Miss", "⚠️ False Alarm", "🔍 Off-Frequency Scan"],
                    range=["#2ec4b6", "#e71d36", "#ff9f1c", "#6c757d"],
                ),
                legend=alt.Legend(title="Scan Outcome"),
            ),
            shape=alt.Shape(
                "Category:N",
                scale=alt.Scale(
                    domain=["🎯 Scan Hit", "❌ Scan Miss", "⚠️ False Alarm", "🔍 Off-Frequency Scan"],
                    range=["circle", "cross", "triangle", "diamond"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Time Step:Q"),
                alt.Tooltip("Category:N"),
                alt.Tooltip("Receiver Freq (MHz):Q", format=".1f"),
                alt.Tooltip("Diagnostic:N"),
            ],
        )
    )

    chart = (rx_bands + emitter_points + rx_outcomes).properties(
        title="Time × Frequency RF Spectrum & Receiver Scan Window",
        height=height,
    ).interactive()

    return chart


def build_arm_stats_dataframe(scheduler_state: dict[str, Any]) -> pd.DataFrame:
    """Extracts scheduler arm diagnostics across stationary and non-stationary bandits."""
    arm_stats = scheduler_state.get("arm_stats", [])
    if not arm_stats:
        return pd.DataFrame()

    rows = []
    for arm in arm_stats:
        freq = arm.get("frequency_hz", 0.0)
        pulls = arm.get("count", arm.get("pulls", 0))
        val = arm.get("value", 0.0)
        prob = arm.get("probability", 0.0)
        bonus = arm.get("bonus", 0.0)
        ub = arm.get("upper_bound", 0.0)
        alpha = arm.get("alpha")
        beta = arm.get("beta")

        row = {
            "Band (MHz)": f"{freq / 1e6:.1f}",
            "Center Freq (Hz)": freq,
            "Scans / Pulls": pulls,
            "Mean Reward / Prob": f"{val:.3f}",
            "Exploration Bonus": f"{bonus:.3f}" if bonus else "0.000",
            "Upper Bound (UCB)": f"{ub:.3f}" if ub else "—",
        }
        if alpha is not None and beta is not None:
            row["Beta Alpha"] = f"{alpha:.2f}"
            row["Beta Beta"] = f"{beta:.2f}"
        rows.append(row)

    return pd.DataFrame(rows)
