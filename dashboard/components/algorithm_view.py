from __future__ import annotations

from typing import Any
import streamlit as st
import pandas as pd
from dashboard.simulation_runner import ALGORITHM_DESCRIPTIONS


def render_algorithm_view(sim_data: dict[str, Any]) -> None:
    """Renders detailed algorithm efficiency metrics and time-series performance graphs."""
    time_df = sim_data["time_df"]
    scheduler_name = sim_data["scheduler_name"]

    st.subheader(f"🧠 Algorithm Efficiency: {scheduler_name.upper()}")
    description = ALGORITHM_DESCRIPTIONS.get(scheduler_name.lower(), "RF band scan scheduler.")
    st.markdown(f"**Description**: {description}")

    if time_df.empty:
        st.info("No time-series data available.")
        return

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Cumulative Reward Trajectory")
        st.caption("Measures accumulated value over time; steeper curves indicate higher algorithm efficiency.")
        st.line_chart(
            time_df,
            x="Time Step",
            y="Cumulative Reward",
            color="#00b4d8",
            use_container_width=True,
        )

    with col2:
        st.markdown("#### 🎯 Rolling Hit Rate Convergence")
        st.caption("Rolling 20-step hit rate showing learning efficiency and intercept stabilization.")
        st.line_chart(
            time_df,
            x="Time Step",
            y="Rolling Hit Rate",
            color="#2a9d8f",
            use_container_width=True,
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### ⚡ Receiver Tuning Frequency Over Time (MHz)")
        st.caption("Illustrates the search strategy (sweep, random jump, or bandit exploitation).")
        st.line_chart(
            time_df,
            x="Time Step",
            y="Receiver Freq (MHz)",
            color="#e76f51",
            use_container_width=True,
        )

    with col4:
        st.markdown("#### 🎲 Instantaneous Step Reward Distribution")
        st.caption("Step-by-step rewards received based on detection/miss outcomes.")
        st.bar_chart(
            time_df.tail(60),
            x="Time Step",
            y="Step Reward",
            color="#7209b7",
            use_container_width=True,
        )
