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
from dashboard.components.spectral_view import render_spectral_view
from dashboard.components.metrics_view import render_metrics_overview
from dashboard.components.why_missed_view import render_why_missed_view
from dashboard.components.intelligence_view import render_intelligence_view

st.set_page_config(
    page_title="SIH26055: Smart Scan Strategy for Electronic Warfare V2.2",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.title("📡 SIH26055: Smart Scan Strategy for Electronic Warfare (V2.2)")
st.markdown("""
**Cognitive Radio & Electronic Support Receiver Simulation System**  
Empirical multi-armed bandit, contextual transition learning, and explainable decision telemetry under agile RF hopping environments.
""")

# Sidebar Configuration
st.sidebar.header("⚙️ Experiment Controls")

# Scenario Selector
scenarios = list_available_scenarios()
priority_scenarios = [
    "deterministic_hopping.yaml",
    "random_hopping.yaml",
    "stationary_fixed.yaml",
    "periodic_time.yaml",
    "mixed_environment.yaml",
]
ordered_scenarios = [s for s in priority_scenarios if s in scenarios] + [s for s in scenarios if s not in priority_scenarios]
default_sc_idx = 0
selected_scenario_name = st.sidebar.selectbox("Scenario Preset", ordered_scenarios, index=default_sc_idx)
current_scenario = load_scenario_file(selected_scenario_name)

# Algorithm Selector
selected_algorithm = st.sidebar.selectbox(
    "Scan Scheduler Algorithm",
    AVAILABLE_ALGORITHMS,
    index=6,  # Default to context_aware
    format_func=lambda x: f"{x.upper()} ({x})",
)

steps = st.sidebar.slider("Simulation Steps", min_value=50, max_value=2000, value=300, step=50)
seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=999999, step=1)

run_button = st.sidebar.button("▶️ Run / Refresh Simulation", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.markdown(f"**Algorithm Description**:\n{ALGORITHM_DESCRIPTIONS.get(selected_algorithm, '')}")

sim_cache_key = f"{selected_scenario_name}_{selected_algorithm}_{steps}_{seed}"

if run_button or "current_sim_data" not in st.session_state or st.session_state.get("last_sim_key") != sim_cache_key:
    if current_scenario is not None:
        with st.spinner(f"Running simulation with {selected_algorithm.upper()}..."):
            sim_data = run_single_simulation(
                scenario=current_scenario,
                scheduler_name=selected_algorithm,
                steps=steps,
                seed=seed,
            )
            sim_data["scenario_name"] = selected_scenario_name
            st.session_state["current_sim_data"] = sim_data
            st.session_state["last_sim_key"] = sim_cache_key

# 4 Main Explained Tabs (Section 15)
tab_rf, tab_perf, tab_missed, tab_intel = st.tabs([
    "🌊 Live RF / Smart Scan",
    "📊 Performance",
    "❓ Why Did We Miss?",
    "🧠 Algorithm Intelligence",
])

if "current_sim_data" in st.session_state:
    sim_data = st.session_state["current_sim_data"]

    with tab_rf:
        render_spectral_view(sim_data)

    with tab_perf:
        render_metrics_overview(sim_data)

    with tab_missed:
        render_why_missed_view(sim_data)

    with tab_intel:
        render_intelligence_view(sim_data)
else:
    st.info("Configure parameters in the sidebar and click **Run Simulation**.")
