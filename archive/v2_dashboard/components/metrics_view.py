from __future__ import annotations

from typing import Any
import streamlit as st
import pandas as pd
from dashboard.simulation_runner import run_benchmark_comparison, AVAILABLE_ALGORITHMS


def render_metrics_overview(sim_data: dict[str, Any]) -> None:
    """Renders Tab 2: Performance

    Features:
    - Explicitly scoped KPIs (Hop-level for agile emitters, Burst/Episode for stationary/periodic).
    - Avoids naked misleading '100% Interception' numbers without scope.
    - Mathematical Identity Verification Check: IR == Coverage * DGC.
    - Full 7-Algorithm Comparative Scoreboard on the identical frozen RF realization.
    - Detector vs Scheduler separation.
    - Opportunity Ledger.
    """
    snapshot = sim_data["snapshot"]
    opps_df = sim_data.get("opportunities_df")
    emitter_df = sim_data.get("emitter_stats_df")
    scenario = sim_data.get("scenario")
    steps = sim_data.get("steps", 200)
    seed = scenario.get("simulation", {}).get("seed", 42) if scenario else 42
    scope = getattr(snapshot, "metric_scope", "HOP")

    st.subheader("📊 Rigorous Performance Evaluation & Scoped Metrics")
    st.markdown("""
    All metrics explicitly state their **evaluation scope** (`HOP`, `BURST`, `EPISODE`, or `STEP`)
    to prevent false aggregate assumptions.
    """)

    # 1. Primary Scoped Scheduler KPIs
    st.markdown(f"### 📡 Primary Scheduler Metrics [Scope: **{scope}**]")

    if scope == "HOP":
        st.info("💡 **Agile Hopping Mode**: KPIs are evaluated across discrete hopping/dwell opportunities.")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric(
                "Hop Interception Ratio",
                f"{snapshot.hop_interception_ratio:.2%}",
                help="Intercepted Hops / Total Hops (Primary Agile Metric)",
            )
        with k2:
            st.metric(
                "Hop Coverage Ratio",
                f"{snapshot.hop_coverage_ratio:.2%}",
                help="Hops where receiver instantaneous bandwidth covered the active frequency.",
            )
        with k3:
            st.metric(
                "Detection Given Coverage",
                f"{snapshot.detection_given_coverage:.2%}",
                help="P(Detected | Covered) = Intercepted Hops / Covered Hops.",
            )
        with k4:
            st.metric(
                "Step Coverage Ratio",
                f"{snapshot.step_coverage_ratio:.2%}",
                help="Overlapping receiver steps / Total active transmitting emitter steps.",
            )
        with k5:
            avg_delay = f"{snapshot.average_intercept_time:.2f} steps" if snapshot.average_intercept_time is not None else "N/A"
            st.metric("Avg Intercept Delay", avg_delay)

        with st.expander("🔍 View Macro Episode Statistics (For Comparison)"):
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric("Episode Interception Ratio", f"{snapshot.episode_interception_ratio:.2%}")
            with e2:
                st.metric("Episode Coverage Ratio", f"{snapshot.episode_coverage_ratio:.2%}")
            with e3:
                st.metric("Total Macro Episodes", snapshot.total_episodes)
            with e4:
                st.metric("Total Discrete Hops", snapshot.total_hops)

    else:
        st.info(f"💡 **{scope} Mode**: KPIs are evaluated across continuous/burst transmission opportunities.")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric(
                "Opportunity Coverage",
                f"{snapshot.opportunity_coverage:.2%}",
                help="Opportunities where receiver tuned to the correct band at least once.",
            )
        with k2:
            st.metric(
                "Detection Given Coverage",
                f"{snapshot.detection_given_coverage:.2%}",
                help="P(Detected | Covered) = Intercepted / Covered Opportunities.",
            )
        with k3:
            st.metric(
                "Interception Ratio",
                f"{snapshot.interception_ratio:.2%}",
                help="Coverage × Detection Given Coverage.",
            )
        with k4:
            avg_delay = f"{snapshot.average_intercept_time:.2f} steps" if snapshot.average_intercept_time is not None else "N/A"
            st.metric("Avg Intercept Delay", avg_delay)

    # Mathematical Identity Banner
    st.markdown(f"""
    > **Mathematical Identity Check**:  
    > $\\text{{Interception Ratio}} \\approx \\text{{Coverage}} \\times \\text{{Detection Given Coverage}}$  
    > **Status**: `{'✅ VERIFIED (|LHS - RHS| < 1e-5)' if snapshot.identity_verified else '❌ VIOLATION'}`
    """)

    st.divider()

    # 2. Detector Performance
    st.markdown("### 🎯 Detector Hardware Performance")
    st.caption("Measures how reliably the detector triggers when an active signal is in-band.")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Probability of Detection (Pd)", f"{snapshot.probability_of_detection:.2%}", help="Hits / (Hits + Misses)")
    with d2:
        st.metric("Probability of False Alarm (Pfa)", f"{snapshot.probability_of_false_alarm:.2%}", help="False Alarms / (FA + CR)")
    with d3:
        st.metric("Miss Rate", f"{snapshot.miss_rate:.2%}", help="1 - Pd")
    with d4:
        st.metric("Correct Rejection Rate", f"{snapshot.correct_rejection_rate:.2%}", help="1 - Pfa")

    st.divider()

    # 3. Full 7-Algorithm Comparative Scoreboard
    st.markdown("### ⚔️ Algorithm Comparison Scoreboard (Identical Frozen Realization)")
    st.caption("All 7 algorithms evaluated against the exact same emitter realization, hop sequences, and channel conditions.")

    bench_cache_key = f"bench_{sim_data.get('scenario_name', 'default')}_{steps}_{seed}"
    if st.button("🔄 Compute / Refresh 7-Algorithm Comparison", type="secondary") or bench_cache_key not in st.session_state:
        with st.spinner("Benchmarking all 7 schedulers on identical realization..."):
            bench_res = run_benchmark_comparison(
                scenario=scenario,
                algorithms=AVAILABLE_ALGORITHMS,
                seeds=[seed],
                steps=steps,
            )
            st.session_state[bench_cache_key] = bench_res

    if bench_cache_key in st.session_state:
        bench_data = st.session_state[bench_cache_key]
        runs_df = bench_data.get("runs_df", pd.DataFrame())
        if not runs_df.empty:
            scoreboard = runs_df[[
                "algorithm",
                "hop_coverage_ratio",
                "hop_interception_ratio",
                "detection_given_coverage",
                "avg_intercept_time",
                "prediction_accuracy",
                "hits",
            ]].copy()

            scoreboard.columns = [
                "Algorithm",
                "Hop Coverage",
                "Hop Interception",
                "Detection Given Coverage",
                "Avg Intercept Delay",
                "Prediction Accuracy",
                "Hits",
            ]

            scoreboard["Hop Coverage"] = scoreboard["Hop Coverage"].apply(lambda v: f"{v:.2%}" if v is not None else "—")
            scoreboard["Hop Interception"] = scoreboard["Hop Interception"].apply(lambda v: f"{v:.2%}" if v is not None else "—")
            scoreboard["Detection Given Coverage"] = scoreboard["Detection Given Coverage"].apply(lambda v: f"{v:.2%}" if v is not None else "—")
            scoreboard["Prediction Accuracy"] = scoreboard["Prediction Accuracy"].apply(lambda v: f"{v:.2%}" if v is not None else "—")
            scoreboard["Avg Intercept Delay"] = scoreboard["Avg Intercept Delay"].apply(lambda v: f"{v:.2f}" if v is not None else "—")

            st.dataframe(scoreboard, use_container_width=True, hide_index=True)

    st.divider()

    # 4. Emitter Status Table
    st.markdown("### 🛰️ Emitter Telemetry & Detection Status")
    if emitter_df is not None and not emitter_df.empty:
        st.dataframe(emitter_df, use_container_width=True, hide_index=True)
