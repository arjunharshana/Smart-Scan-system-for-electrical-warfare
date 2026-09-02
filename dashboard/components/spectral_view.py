from __future__ import annotations

from typing import Any
import streamlit as st
import pandas as pd
from dashboard.utils import format_frequency, format_bandwidth


def render_spectral_view(sim_data: dict[str, Any]) -> None:
    """Renders the RF spectrum view, waterfall spectrogram, and receiver hardware profile."""
    env = sim_data["env"]
    waterfall_df = sim_data["waterfall_df"]
    spectrum = env.spectrum
    receiver = env.receiver

    st.subheader("🌊 RF Spectrum & Waterfall Spectrogram")
    st.markdown("""
    This view maps the temporal and spectral behavior of the RF environment.
    Points represent active signal transmissions from emitters and the instantaneous tuned bands probed by the receiver.
    """)

    # Hardware & Spectrum Info Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**Spectrum Range**\n\n{format_frequency(spectrum['min_frequency_hz'])} — {format_frequency(spectrum['max_frequency_hz'])}")
    with col2:
        st.info(f"**Receiver Bandwidth**\n\n{format_bandwidth(receiver.instantaneous_bandwidth_hz)}")
    with col3:
        st.info(f"**Receiver Sensitivity**\n\n`{receiver.sensitivity_dbm:.1f} dBm`")
    with col4:
        st.info(f"**Noise Floor**\n\n`{receiver.noise_floor_dbm:.1f} dBm`")

    st.divider()

    if not waterfall_df.empty:
        st.markdown("#### 🛰️ Waterfall Spectrogram (Time vs Frequency)")
        
        # Option to filter view
        event_types = sorted(waterfall_df["Category"].unique())
        selected_categories = st.multiselect(
            "Filter Waterfall Layers",
            options=event_types,
            default=event_types,
        )

        filtered_df = waterfall_df[waterfall_df["Category"].isin(selected_categories)]

        if not filtered_df.empty:
            st.scatter_chart(
                filtered_df,
                x="Frequency (MHz)",
                y="Time Step",
                color="Category",
                use_container_width=True,
            )
        else:
            st.warning("No data matches the selected filters.")
            
        with st.expander("🔍 View Raw Waterfall Telemetry Table"):
            st.dataframe(waterfall_df.tail(100), use_container_width=True)
    else:
        st.info("No waterfall events recorded.")
