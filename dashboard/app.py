from __future__ import annotations

import os
import sys
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.utils import list_available_scenarios, load_scenario_file
from dashboard.simulation_runner import (
    run_single_simulation,
    AVAILABLE_ALGORITHMS,
    ALGORITHM_DESCRIPTIONS,
)
from dashboard.components.metrics_view import render_metrics_overview
from dashboard.components.spectral_view import render_spectral_view
from dashboard.components.algorithm_view import render_algorithm_view
from dashboard.components.bandit_view import render_bandit_view
from dashboard.components.comparison_view import render_comparison_view
from dashboard.components.custom_scenario_view import (
    render_custom_scenario_view,
    get_custom_scenario,
)

st.set_page_config(
    page_title="RF Environment & Algorithm Intelligence Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("📡 Cognitive RF Environment & Algorithm Performance Intelligence")
st.markdown("""
Interactive telemetry analysis, algorithm benchmarking, and RF environment simulation platform for **Cognitive Radio & Electronic Support (ES)** scan schedulers.
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Simulation Settings")

scenario_source = st.sidebar.radio(
    "Scenario Configuration Source",
    ["Preset Scenarios", "Custom Sandbox Builder"],
    index=0,
)

if scenario_source == "Preset Scenarios":
    scenarios = list_available_scenarios()
    selected_scenario_name = st.sidebar.selectbox("Select Scenario Preset", scenarios)
    current_scenario = load_scenario_file(selected_scenario_name)
else:
    selected_scenario_name = "Custom Sandbox Scenario"
    current_scenario = get_custom_scenario()

selected_algorithm = st.sidebar.selectbox(
    "Scheduler Algorithm",
    AVAILABLE_ALGORITHMS,
    index=2,  # Default to UCB1
    format_func=lambda x: f"{x.upper()} ({x})",
)

steps = st.sidebar.slider("Simulation Steps", min_value=20, max_value=2000, value=200, step=20)
seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=999999, step=1)

run_button = st.sidebar.button("▶️ Run / Refresh Simulation", type="primary", use_container_width=True)

# Determine if we should compute simulation
sim_cache_key = f"{scenario_source}_{selected_scenario_name}_{selected_algorithm}_{steps}_{seed}"

if run_button or "current_sim_data" not in st.session_state or st.session_state.get("last_sim_key") != sim_cache_key:
    if current_scenario is not None:
        with st.spinner(f"Running simulation with {selected_algorithm.upper()}..."):
            sim_data = run_single_simulation(
                scenario=current_scenario,
                scheduler_name=selected_algorithm,
                steps=steps,
                seed=seed,
            )
            st.session_state["current_sim_data"] = sim_data
            st.session_state["last_sim_key"] = sim_cache_key

# Sequential Tab Rendering (guaranteed order & no blank DOM mount issues)
tab_kpi, tab_waterfall, tab_algo, tab_bandit, tab_bench, tab_sandbox = st.tabs([
    "📊 KPI & Detection Metrics",
    "🌊 Waterfall Spectrogram",
    "📈 Algorithm Dynamics",
    "🎰 Bandit Arm Stats",
    "⚔️ Benchmark Comparison",
    "🛠️ Scenario Sandbox",
])

if "current_sim_data" in st.session_state:
    sim_data = st.session_state["current_sim_data"]

    with tab_kpi:
        render_metrics_overview(sim_data)

    with tab_waterfall:
        render_spectral_view(sim_data)

    with tab_algo:
        render_algorithm_view(sim_data)

    with tab_bandit:
        render_bandit_view(sim_data)

    with tab_bench:
        render_comparison_view(
            scenario=current_scenario if current_scenario else sim_data["scenario"],
            steps=steps,
            seed=seed,
        )

    with tab_sandbox:
        render_custom_scenario_view()
else:
    st.info("Configure simulation parameters in the sidebar and click **Run Simulation**.")
