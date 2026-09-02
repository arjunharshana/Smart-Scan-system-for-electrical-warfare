from __future__ import annotations

from typing import Any
import streamlit as st
import pandas as pd
from dashboard.simulation_runner import run_benchmark_comparison, AVAILABLE_ALGORITHMS


def render_comparison_view(scenario: dict[str, Any], steps: int, seed: int) -> None:
    """Renders the Multi-Algorithm Benchmark Comparison suite."""
    st.subheader("⚔️ Multi-Algorithm Efficiency Benchmark")
    st.markdown("""
    Compare **Sequential**, **Random**, **UCB1**, and **Thompson Sampling** algorithms side-by-side under identical RF scenario conditions and random seed.
    """)

    selected_algorithms = st.multiselect(
        "Select Algorithms to Benchmark",
        options=AVAILABLE_ALGORITHMS,
        default=AVAILABLE_ALGORITHMS,
    )

    if not selected_algorithms:
        st.warning("Please select at least one algorithm to run the benchmark.")
        return

    if st.button("🚀 Run Full Multi-Algorithm Benchmark", type="primary"):
        with st.spinner("Executing multi-algorithm benchmarks in parallel..."):
            benchmark_data = run_benchmark_comparison(
                scenario=scenario,
                algorithms=selected_algorithms,
                steps=steps,
                seed=seed,
            )

        st.session_state["benchmark_data"] = benchmark_data

    if "benchmark_data" in st.session_state:
        benchmark_data = st.session_state["benchmark_data"]
        summary_df = benchmark_data["summary_df"]
        trajectory_df = benchmark_data["trajectory_df"]

        # Winner callout
        winner = summary_df.iloc[0]["Algorithm"]
        winner_reward = summary_df.iloc[0]["Total Cumulative Reward"]
        st.success(f"🏆 **Benchmark Winner**: **{winner}** achieved the highest cumulative reward of **{winner_reward:.1f}** over {steps} steps!")

        st.markdown("### 📋 Comparative Performance Scoreboard")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.divider()

        # Multi-line Cumulative Reward chart
        st.markdown("### 📈 Cumulative Reward Trajectory Comparison")
        st.caption("Illustrates how quickly each algorithm learns and accumulates detections over time.")
        
        pivot_rewards = trajectory_df.pivot(index="Time Step", columns="Algorithm", values="Cumulative Reward")
        st.line_chart(pivot_rewards, use_container_width=True)

        # Rolling hit rate comparison
        st.markdown("### 🎯 Rolling Hit Rate Comparison (Learning Speed)")
        st.caption("Shows instantaneous detection consistency and convergence over time.")
        pivot_hit_rate = trajectory_df.pivot(index="Time Step", columns="Algorithm", values="Rolling Hit Rate")
        st.line_chart(pivot_hit_rate, use_container_width=True)

        st.divider()

        # Comparative Bar Charts
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎯 Total Hits Comparison")
            st.bar_chart(summary_df.set_index("Algorithm")["Hits"], color="#06d6a0", use_container_width=True)

        with c2:
            st.markdown("#### 📡 Interception Ratio Comparison")
            st.bar_chart(summary_df.set_index("Algorithm")["Interception Ratio"], color="#118ab2", use_container_width=True)
