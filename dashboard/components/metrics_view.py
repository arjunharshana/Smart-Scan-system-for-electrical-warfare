from __future__ import annotations

from typing import Any
import streamlit as st
import pandas as pd


def render_metrics_overview(sim_data: dict[str, Any]) -> None:
    """Renders high-level statistical KPI cards and outcome breakdown."""
    snapshot = sim_data["snapshot"]

    st.subheader("📊 Executive Summary & Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Scans", snapshot.total_scans)
    with col2:
        st.metric("Hits (Detected)", snapshot.hits, delta=f"{snapshot.hits / max(snapshot.total_scans, 1):.1%} rate")
    with col3:
        st.metric("Misses", snapshot.misses)
    with col4:
        st.metric("False Alarms", snapshot.false_alarms)
    with col5:
        st.metric("Correct Rejections", snapshot.correct_rejections)

    col6, col7, col8, col9, col10 = st.columns(5)
    with col6:
        st.metric("Probability of Detection (Pd)", f"{snapshot.probability_of_detection:.2%}")
    with col7:
        st.metric("False Alarm Rate (Pfa)", f"{snapshot.probability_of_false_alarm:.2%}")
    with col8:
        st.metric("Interception Ratio", f"{snapshot.interception_ratio:.2%}")
    with col9:
        st.metric("Average Reward / Step", f"{snapshot.average_reward:.3f}")
    with col10:
        st.metric("Prediction Accuracy", f"{snapshot.prediction_accuracy:.2%}")

    st.divider()

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### 🎯 Scan Outcome Distribution")
        outcomes_data = pd.DataFrame([
            {"Outcome": "Hits (True Positives)", "Count": snapshot.hits},
            {"Outcome": "Misses (False Negatives)", "Count": snapshot.misses},
            {"Outcome": "False Alarms (False Positives)", "Count": snapshot.false_alarms},
            {"Outcome": "Correct Rejections (True Negatives)", "Count": snapshot.correct_rejections},
        ])
        st.bar_chart(outcomes_data.set_index("Outcome"), color="#2b7fff", use_container_width=True)

    with c2:
        st.markdown("#### 📡 Emitter Intercept & Discovery Status")
        emitter_df = sim_data.get("emitter_stats_df")
        if emitter_df is not None and not emitter_df.empty:
            st.dataframe(emitter_df, use_container_width=True, hide_index=True)
            if snapshot.average_intercept_time is not None:
                st.caption(f"⚡ **Average Time to First Intercept**: `{snapshot.average_intercept_time:.1f}` simulation steps across discovered emitters.")
        else:
            st.info("No emitter details available.")
