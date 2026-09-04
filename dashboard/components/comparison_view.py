from __future__ import annotations

from typing import Any
import streamlit as st
import pandas as pd
from dashboard.simulation_runner import (
    run_benchmark_comparison,
    AVAILABLE_ALGORITHMS,
    ALGORITHM_DESCRIPTIONS,
)


def render_comparison_view(scenario: dict[str, Any], steps: int, seed: int) -> None:
    """Renders Tab 3: Algorithm Comparison - Multi-Seed Benchmark Suite with Mean ± Std."""
    st.subheader("⚔️ Multi-Algorithm & Multi-Seed Comparative Benchmark")
    st.markdown("""
    Evaluates scan schedulers across identical frozen RF environment realizations.
    Statistical results display **Mean ± Standard Deviation** across runs to avoid single-seed bias.
    """)

    # Controls
    col_c1, col_c2 = st.columns([3, 1])
    with col_c1:
        selected_algorithms = st.multiselect(
            "Select Algorithms to Benchmark",
            options=AVAILABLE_ALGORITHMS,
            default=AVAILABLE_ALGORITHMS,
            format_func=lambda x: f"{x.upper()} ({ALGORITHM_DESCRIPTIONS.get(x, '').split(':')[0]})",
        )
    with col_c2:
        num_seeds = st.selectbox("Number of Seeds", [1, 3, 5, 10], index=1)

    if not selected_algorithms:
        st.warning("Please select at least one algorithm.")
        return

    benchmark_button = st.button("🚀 Run Multi-Seed Benchmark", type="primary", use_container_width=True)

    seeds_list = list(range(1, num_seeds + 1))
    bench_cache_key = f"bench_{selected_algorithms}_{num_seeds}_{steps}_{scenario.get('simulation', {}).get('seed')}"

    if benchmark_button:
        with st.spinner(f"Running benchmark across {num_seeds} seeds for {len(selected_algorithms)} algorithms..."):
            benchmark_data = run_benchmark_comparison(
                scenario=scenario,
                algorithms=selected_algorithms,
                seeds=seeds_list,
                steps=steps,
            )
            st.session_state["benchmark_data"] = benchmark_data
            st.session_state["bench_key"] = bench_cache_key

    if "benchmark_data" in st.session_state:
        benchmark_data = st.session_state["benchmark_data"]
        summary_df = benchmark_data["summary_df"]
        runs_df = benchmark_data["runs_df"]

        st.markdown(f"### 📋 Benchmark Scoreboard ({num_seeds} Seeds × {steps} Steps)")
        st.caption("Values formatted as Mean ± Std Dev. Best values per metric should be compared scientifically.")

        display_cols = [
            "Algorithm",
            "interception_ratio",
            "opportunity_coverage",
            "detection_given_coverage",
            "avg_intercept_time",
            "prediction_accuracy",
            "Pd",
            "Pfa",
            "avg_reward",
        ]
        valid_cols = [c for c in display_cols if c in summary_df.columns]
        st.dataframe(summary_df[valid_cols], use_container_width=True, hide_index=True)

        st.divider()

        # Comparative Bar Charts
        st.markdown("### 📊 Side-by-Side Performance Comparison")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### 🎯 Interception Ratio")
            st.caption("Opportunity-based interception rate.")
            chart_df = summary_df.set_index("Algorithm")["interception_ratio_mean"]
            st.bar_chart(chart_df, color="#118ab2", use_container_width=True)

        with c2:
            st.markdown("#### 📡 Opportunity Coverage")
            st.caption("Fraction of episodes where receiver scanned in-band.")
            chart_df = summary_df.set_index("Algorithm")["opportunity_coverage_mean"]
            st.bar_chart(chart_df, color="#06d6a0", use_container_width=True)

        with c3:
            st.markdown("#### ⚡ Average Intercept Delay")
            st.caption("Steps to first detection (lower is better).")
            chart_df = summary_df.set_index("Algorithm")["avg_intercept_time_mean"]
            st.bar_chart(chart_df, color="#ffd166", use_container_width=True)

        with st.expander("🔍 View Raw Per-Seed Benchmark Runs Data"):
            st.dataframe(runs_df, use_container_width=True, hide_index=True)
