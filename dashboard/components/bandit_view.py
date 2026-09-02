from __future__ import annotations

from typing import Any
import streamlit as st
import pandas as pd


def render_bandit_view(sim_data: dict[str, Any]) -> None:
    """Renders Multi-Armed Bandit arm selection statistics and exploration vs exploitation analysis."""
    arm_df = sim_data["arm_df"]
    scheduler_name = sim_data["scheduler_name"].lower()

    st.subheader(f"🎰 Multi-Armed Bandit & Frequency Allocation Dynamics")

    if arm_df.empty:
        st.info("No specific bandit arm statistics available for this scheduler.")
        return

    total_pulls = arm_df["Scans / Pulls"].sum()
    top_arm = arm_df.sort_values(by="Scans / Pulls", ascending=False).iloc[0]

    # Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Band Scans", total_pulls)
    with c2:
        st.metric("Most Scanned Band", f"{top_arm['Frequency (MHz)']} MHz", f"{top_arm['Scans / Pulls']} scans")
    with c3:
        exploitation_ratio = (top_arm["Scans / Pulls"] / max(total_pulls, 1))
        st.metric("Top-Band Focus Ratio", f"{exploitation_ratio:.1%}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Band Scan / Pull Distribution")
        st.caption("Shows how frequently the scheduler allocated scanning attention across channels.")
        st.bar_chart(
            arm_df.set_index("Frequency (MHz)")["Scans / Pulls"],
            color="#4361ee",
            use_container_width=True,
        )

    with col2:
        st.markdown("#### 🏆 Mean Reward per Frequency Band")
        st.caption("Empirical average reward discovered per frequency channel.")
        st.bar_chart(
            arm_df.set_index("Frequency (MHz)")["Mean Reward"],
            color="#4cc9f0",
            use_container_width=True,
        )

    with st.expander("📋 Detailed Channel Band Arm Statistics Table"):
        st.dataframe(arm_df, use_container_width=True, hide_index=True)
