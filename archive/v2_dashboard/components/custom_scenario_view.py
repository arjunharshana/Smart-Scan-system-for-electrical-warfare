from __future__ import annotations

from typing import Any
import streamlit as st
import yaml


def init_custom_scenario_state() -> None:
    """Initializes session state for custom scenario if not present."""
    if "custom_emitters" not in st.session_state:
        st.session_state["custom_emitters"] = [
            {
                "id": "E_RADAR_01",
                "type": "radar",
                "frequency_behavior": {"type": "fixed", "frequency_hz": 350000000},
                "time_behavior": {"type": "continuous"},
                "power_dbm": -20.0,
                "bandwidth_hz": 2000000.0,
            },
            {
                "id": "E_COMM_02",
                "type": "communication",
                "frequency_behavior": {
                    "type": "hopping",
                    "frequencies_hz": [200000000, 400000000, 700000000],
                    "mode": "sequential",
                    "dwell_steps": 2,
                },
                "time_behavior": {"type": "burst", "burst_duration": 4, "interval": 15},
                "power_dbm": -25.0,
                "bandwidth_hz": 1000000.0,
            },
        ]
    if "custom_rx_params" not in st.session_state:
        st.session_state["custom_rx_params"] = {
            "min_freq_mhz": 100.0,
            "max_freq_mhz": 1000.0,
            "rx_bw_mhz": 20.0,
            "sensitivity_dbm": -90.0,
            "noise_floor_dbm": -100.0,
            "det_thresh_db": 6.0,
        }


def get_custom_scenario() -> dict[str, Any]:
    """Generates the scenario dictionary from current custom session state."""
    init_custom_scenario_state()
    rx = st.session_state["custom_rx_params"]
    emitters = st.session_state["custom_emitters"]

    return {
        "simulation": {"total_time_steps": 200, "time_step_ms": 10, "seed": 42},
        "spectrum": {
            "min_frequency_hz": float(rx["min_freq_mhz"]) * 1e6,
            "max_frequency_hz": float(rx["max_freq_mhz"]) * 1e6,
        },
        "receiver": {
            "instantaneous_bandwidth_hz": float(rx["rx_bw_mhz"]) * 1e6,
            "sensitivity_dbm": float(rx["sensitivity_dbm"]),
            "noise_floor_dbm": float(rx["noise_floor_dbm"]),
            "detection_threshold_db": float(rx["det_thresh_db"]),
            "tuning_time_ms": 0,
        },
        "detector": {
            "p_detection": 0.95,
            "p_false_alarm": 0.02,
        },
        "scheduler": {"type": "ucb1"},
        "emitters": emitters,
    }


def render_custom_scenario_view() -> None:
    """Renders the custom scenario builder and configuration GUI."""
    init_custom_scenario_state()

    st.subheader("🛠️ Interactive Scenario & Hardware Sandbox")
    st.markdown("""
    Customize RF spectrum boundaries, receiver hardware specifications, and manage active emitters.
    """)

    rx = st.session_state["custom_rx_params"]

    # Spectrum and Receiver config
    with st.expander("📡 Spectrum & Receiver Hardware Configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            rx["min_freq_mhz"] = st.number_input("Min Spectrum Frequency (MHz)", value=float(rx["min_freq_mhz"]), step=10.0, key="cfg_min_freq")
            rx["max_freq_mhz"] = st.number_input("Max Spectrum Frequency (MHz)", value=float(rx["max_freq_mhz"]), step=50.0, key="cfg_max_freq")
            rx["rx_bw_mhz"] = st.number_input("Receiver Instantaneous Bandwidth (MHz)", value=float(rx["rx_bw_mhz"]), step=5.0, key="cfg_rx_bw")

        with c2:
            rx["sensitivity_dbm"] = st.number_input("Receiver Sensitivity (dBm)", value=float(rx["sensitivity_dbm"]), step=5.0, key="cfg_sens")
            rx["noise_floor_dbm"] = st.number_input("Noise Floor (dBm)", value=float(rx["noise_floor_dbm"]), step=5.0, key="cfg_noise")
            rx["det_thresh_db"] = st.number_input("Detection Threshold (dB)", value=float(rx["det_thresh_db"]), step=1.0, key="cfg_thresh")

    # Emitter configuration manager
    st.markdown("### 🛰️ Configured Emitters")
    emitters_list = st.session_state["custom_emitters"]

    to_delete = None
    for idx, em in enumerate(emitters_list):
        with st.expander(f"Emitter {idx+1}: `{em['id']}` ({em['type'].upper()})", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                em["id"] = st.text_input(f"Emitter ID", value=em["id"], key=f"em_id_{idx}")
                em["type"] = st.selectbox(f"Type", ["radar", "communication"], index=0 if em["type"]=="radar" else 1, key=f"em_type_{idx}")
                em["power_dbm"] = st.number_input(f"Power (dBm)", value=float(em.get("power_dbm", -20)), key=f"em_pwr_{idx}")
            with col_b:
                cur_freq_type = em["frequency_behavior"].get("type", "fixed")
                freq_options = ["fixed", "hopping", "sweep", "random"]
                freq_idx = freq_options.index(cur_freq_type) if cur_freq_type in freq_options else 0
                em["frequency_behavior"]["type"] = st.selectbox(
                    f"Frequency Behavior",
                    freq_options,
                    index=freq_idx,
                    key=f"em_freq_type_{idx}",
                )

                cur_time_type = em["time_behavior"].get("type", "continuous")
                time_options = ["continuous", "periodic", "burst", "intermittent"]
                time_idx = time_options.index(cur_time_type) if cur_time_type in time_options else 0
                em["time_behavior"]["type"] = st.selectbox(
                    f"Time Behavior",
                    time_options,
                    index=time_idx,
                    key=f"em_time_type_{idx}",
                )

            if st.button(f"🗑️ Delete Emitter `{em['id']}`", key=f"del_btn_{idx}"):
                to_delete = idx

    if to_delete is not None:
        emitters_list.pop(to_delete)
        st.rerun()

    if st.button("➕ Add New Emitter", key="add_emitter_btn"):
        new_id = f"E_CUSTOM_{len(emitters_list)+1:02d}"
        emitters_list.append({
            "id": new_id,
            "type": "radar",
            "frequency_behavior": {"type": "fixed", "frequency_hz": 500000000},
            "time_behavior": {"type": "continuous"},
            "power_dbm": -20.0,
            "bandwidth_hz": 2000000.0,
        })
        st.rerun()

    scenario_dict = get_custom_scenario()
    with st.expander("📄 View Generated Scenario YAML"):
        st.code(yaml.dump(scenario_dict, default_flow_style=False), language="yaml")
