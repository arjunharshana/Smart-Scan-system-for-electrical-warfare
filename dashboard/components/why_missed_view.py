from __future__ import annotations

from typing import Any
import altair as alt
import streamlit as st
import pandas as pd


def render_why_missed_view(sim_data: dict[str, Any]) -> None:
    """Renders Tab 3: Why Did We Miss?

    Classifies every missed transmission/hop into:
    1. OFF-FREQUENCY: Receiver tuned to a different channel while target was transmitting.
    2. CORRECT FREQUENCY BUT DETECTOR MISS: Receiver overlapped target, but SNR was insufficient or detector failed.
    3. EMITTER NOT TRANSMITTING: Receiver scanned while emitter was silent/off.
    4. NO VALID OPPORTUNITY: Scans outside active emitter spectrum bounds.

    Scientifically demonstrates:
    "Our EW problem is primarily a scheduling problem rather than a detector problem."
    """
    results = sim_data.get("results", [])
    snapshot = sim_data.get("snapshot")
    scheduler_name = sim_data.get("scheduler_name", "scheduler")

    st.subheader(f"❓ Why Did We Miss? — Root Cause Diagnostic Analysis ({scheduler_name.upper()})")
    st.markdown("""
    This diagnostic view decomposes every scan event and missed transmission opportunity into its fundamental physical cause.
    It isolates **scheduling strategy failures** (receiver off-frequency) from **detector hardware limitations** (in-band misses).
    """)

    # 1. Compute Step-Level Diagnostic Counts
    off_frequency_steps = 0
    detector_miss_steps = 0
    emitter_silent_steps = 0
    hits_count = 0
    false_alarms_count = 0

    detailed_records = []
    for r in results:
        t = r.get("timestamp", 0)
        outcome = r.get("outcome", {}).get("outcome", "UNKNOWN")
        rx = r.get("receiver", {})
        rx_f = rx.get("center_frequency_hz", 0.0) / 1e6
        gt_emitters = r.get("ground_truth", {}).get("emitters", [])
        active_emitters = [e for e in gt_emitters if e.get("transmitting", False)]
        diag = r.get("diagnostic_reason", "")

        classification = "UNKNOWN"
        if outcome == "HIT":
            hits_count += 1
            classification = "SUCCESSFUL_INTERCEPT"
        elif outcome == "MISS":
            detector_miss_steps += 1
            classification = "DETECTOR_MISS"
        elif outcome == "FALSE_ALARM":
            false_alarms_count += 1
            classification = "FALSE_ALARM"
        elif outcome == "CORRECT_REJECTION":
            if active_emitters:
                off_frequency_steps += 1
                classification = "OFF_FREQUENCY"
            else:
                emitter_silent_steps += 1
                classification = "EMITTER_IDLE"

        active_freqs = [f"{e['frequency_hz']/1e6:.1f} MHz" for e in active_emitters]
        active_str = ", ".join(active_freqs) if active_freqs else "Silent"

        detailed_records.append({
            "Step": t,
            "Classification": classification,
            "Outcome": outcome,
            "Receiver Freq (MHz)": f"{rx_f:.1f}",
            "Active Emitter Frequencies": active_str,
            "Diagnostic Rationale": diag,
        })

    # Total non-hit steps during transmission
    active_miss_steps = off_frequency_steps + detector_miss_steps
    total_active_tx_steps = active_miss_steps + hits_count

    off_freq_pct = (off_frequency_steps / active_miss_steps * 100.0) if active_miss_steps > 0 else 0.0
    det_miss_pct = (detector_miss_steps / active_miss_steps * 100.0) if active_miss_steps > 0 else 0.0
    other_pct = 0.0

    # 2. Top Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Missed Tx Steps", f"{active_miss_steps}", f"{active_miss_steps}/{max(total_active_tx_steps, 1)} tx steps")
    with c2:
        st.metric("Off-Frequency (Scheduling Loss)", f"{off_freq_pct:.1f}%", f"{off_frequency_steps} steps")
    with c3:
        st.metric("Detector Miss (In-Band SNR)", f"{det_miss_pct:.1f}%", f"{detector_miss_steps} steps")
    with c4:
        st.metric("Idle Channel Scans", f"{emitter_silent_steps}", "Target not transmitting")

    st.divider()

    # 3. EW Problem Significance Banner
    if off_freq_pct >= 80.0:
        st.success(f"""
        🎯 **Core EW Electronic Support Insight**:
        **{off_freq_pct:.1f}%** of all missed opportunities were caused by the receiver scanning **Off-Frequency**,
        while only **{det_miss_pct:.1f}%** were lost to detector sensitivity limits.
        This scientifically demonstrates that **intelligent cognitive scan scheduling** is the dominant operational factor in Electronic Warfare intercept capability.
        """)
    else:
        st.info(f"""
        📊 **Root Cause Breakdown**:
        Off-Frequency Scheduling Loss: **{off_freq_pct:.1f}%** | In-Band Detector Misses: **{det_miss_pct:.1f}%**.
        """)

    col_chart, col_tree = st.columns([1.2, 1])

    with col_chart:
        st.markdown("### 📊 Miss Classification Distribution")
        chart_data = pd.DataFrame([
            {"Root Cause": "Off-Frequency (Wrong Band)", "Count": off_frequency_steps, "Percentage": off_freq_pct},
            {"Root Cause": "Detector Miss (In-Band)", "Count": detector_miss_steps, "Percentage": det_miss_pct},
            {"Root Cause": "Target Silent / Idle", "Count": emitter_silent_steps, "Percentage": 0.0},
        ])

        chart = (
            alt.Chart(chart_data[chart_data["Count"] > 0])
            .mark_bar(cornerRadius=6)
            .encode(
                x=alt.X("Count:Q", title="Number of Occurrences (Steps)"),
                y=alt.Y("Root Cause:N", sort="-x", title="Root Cause"),
                color=alt.Color(
                    "Root Cause:N",
                    scale=alt.Scale(
                        domain=["Off-Frequency (Wrong Band)", "Detector Miss (In-Band)", "Target Silent / Idle"],
                        range=["#e71d36", "#ff9f1c", "#6c757d"],
                    ),
                    legend=None,
                ),
                tooltip=["Root Cause", "Count", alt.Tooltip("Percentage:Q", format=".1f")],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, use_container_width=True)

    with col_tree:
        st.markdown("### 🌲 Hierarchical Miss Decomposition")
        tree_text = f"""
```text
Missed Transmissions
│
├── 🔴 Off-frequency (Scheduling Miss)      {off_freq_pct:>5.1f}%
├── 🟠 Detector miss (SNR / Threshold)       {det_miss_pct:>5.1f}%
└── ⚪ Idle / Outside Opportunity            {other_pct:>5.1f}%
```
"""
        st.markdown(tree_text)
        st.markdown(f"""
        - **Total Active Tx Steps**: `{total_active_tx_steps}`
        - **Successful Hits**: `{hits_count}`
        - **Off-Frequency Scans**: `{off_frequency_steps}`
        - **In-Band Detector Misses**: `{detector_miss_steps}`
        - **False Alarms**: `{false_alarms_count}`
        """)

    st.divider()

    # 4. Filterable Step Diagnostic Log
    st.markdown("### 📋 Step-by-Step Scan Diagnostic Log")
    filter_choice = st.radio(
        "Filter Events:",
        ["All Events", "Off-Frequency Only (Scheduling Misses)", "Detector Misses Only", "Hits Only"],
        horizontal=True,
    )

    df_detail = pd.DataFrame(detailed_records)
    if not df_detail.empty:
        if filter_choice == "Off-Frequency Only (Scheduling Misses)":
            df_show = df_detail[df_detail["Classification"] == "OFF_FREQUENCY"]
        elif filter_choice == "Detector Misses Only":
            df_show = df_detail[df_detail["Classification"] == "DETECTOR_MISS"]
        elif filter_choice == "Hits Only":
            df_show = df_detail[df_detail["Classification"] == "SUCCESSFUL_INTERCEPT"]
        else:
            df_show = df_detail

        st.dataframe(df_show.tail(150), use_container_width=True, hide_index=True)
