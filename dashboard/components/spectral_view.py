from __future__ import annotations

from typing import Any
import streamlit as st
import pandas as pd
from dashboard.utils import (
    format_frequency,
    format_bandwidth,
    build_waterfall_altair_chart,
)


def render_spectral_view(sim_data: dict[str, Any]) -> None:
    """Renders Tab 1: Live RF / Smart Scan

    Features:
    - Top KPI cards (Receiver Freq, Bandwidth, Detection, SNR, Scheduler, Reward).
    - Time × Frequency Waterfall Spectrogram with moving receiver window.
    - Visual legend for RF emitter trajectories, receiver window, and detections.
    - Current Decision Panel ("WHY THIS FREQUENCY?").
    - Replay Mode (step scrubber, play/pause, per-step details).
    - Opportunity Inspector (inspect individual hop/burst lifecycle).
    """
    env = sim_data["env"]
    results = sim_data.get("results", [])
    waterfall_df = sim_data["waterfall_df"]
    snapshot = sim_data["snapshot"]
    spectrum = env.spectrum
    receiver = env.receiver
    scheduler_name = sim_data.get("scheduler_name", "scheduler")
    last_res = results[-1] if results else {}

    rx_state = receiver.get_state()
    obs = last_res.get("observation", {})
    detection = last_res.get("detection", {})
    outcome_dict = last_res.get("outcome", {})
    sched_state = env.scheduler.get_state()

    st.subheader("🌊 Live RF Spectrum & Smart Scan Telemetry")

    # 1. Top KPI Cards (Section 15)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Receiver Frequency", format_frequency(rx_state.center_frequency_hz))
    with c2:
        st.metric("Receiver Bandwidth", format_bandwidth(rx_state.instantaneous_bandwidth_hz))
    with c3:
        det_label = "🎯 DETECTED" if detection.get("detected") else "❌ NO SIGNAL"
        st.metric("Current Detection", det_label)
    with c4:
        snr_val = detection.get("snr_db")
        st.metric("Current SNR", f"{snr_val:.1f} dB" if snr_val is not None else "— dB")
    with c5:
        st.metric("Current Scheduler", scheduler_name.upper())
    with c6:
        rew = outcome_dict.get("reward", 0.0)
        st.metric("Current Reward", f"{rew:+.2f}")

    st.divider()

    # 2. Primary Time x Frequency Waterfall Visualization
    col_chart, col_decision = st.columns([1.6, 1])

    with col_chart:
        st.markdown("### 📡 Time × Frequency RF Spectrum & Moving Scan Window")
        st.markdown(
            "The **shaded band** traces the moving receiver instantaneous bandwidth window "
            r"($[f_c - \frac{BW}{2}, f_c + \frac{BW}{2}]$), points trace emitter hopping trajectories, and symbols indicate scan outcomes."
        )
        if not waterfall_df.empty:
            chart = build_waterfall_altair_chart(waterfall_df, height=420)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No spectrum data recorded.")

        # Visual Legend
        st.markdown("""
        **Visual Legend**:
        - 🟦 **Shaded Cyan Window**: Receiver Instantaneous Bandwidth Window ($f_c \pm 10\text{ MHz}$)
        - 🟣 **Colored Points**: Active Emitter Signal Trajectory
        - 🎯 **Green Dot**: Successful Scan Hit (Intercept)
        - ❌ **Red Cross**: Scan Miss (In-band signal present, but detector missed)
        - ⚠️ **Orange Triangle**: False Alarm (Detector fired in empty band)
        - 🔍 **Grey Diamond**: Off-Frequency Scan (Receiver tuned to vacant channel)
        """)

    with col_decision:
        # 3. Current Decision Panel ("WHY THIS FREQUENCY?")
        st.markdown("### 🧠 Current Decision")
        st.caption("Real-time explanation derived from actual algorithm state.")

        expl = sched_state.explanation or {}
        last_obs_mhz = "—"
        if obs and obs.get("recent_history"):
            last_f = obs["recent_history"].get("last_detected_frequency_hz")
            if last_f:
                last_obs_mhz = f"{last_f / 1e6:.1f} MHz"

        pred_mhz = "—"
        if hasattr(env.scheduler, "predicted_frequency_hz") and env.scheduler.predicted_frequency_hz:
            pred_mhz = f"{env.scheduler.predicted_frequency_hz / 1e6:.1f} MHz"
        elif "predicted_next_mhz" in expl and expl["predicted_next_mhz"]:
            pred_mhz = f"{expl['predicted_next_mhz']:.1f} MHz"

        confidence_str = "—"
        if "transition_probability" in expl:
            confidence_str = f"{expl['transition_probability']:.1%}"

        reason_text = expl.get("reason", "Explanation unavailable for this scheduler.")
        selected_mhz = f"{rx_state.center_frequency_hz / 1e6:.1f} MHz"

        decision_box = f"""
```text
WHY THIS FREQUENCY?

Scheduler:
{scheduler_name.upper()}

Previous observed frequency:
{last_obs_mhz}

Predicted next frequency:
{pred_mhz}

Prediction confidence:
{confidence_str}

Selected receiver frequency:
{selected_mhz}

Reason:
{reason_text}
```
"""
        st.markdown(decision_box)

        st.metric("Latest Scan Outcome", outcome_dict.get("outcome", "NONE"))
        st.info(last_res.get("diagnostic_reason", "No diagnostic reason recorded."))

    st.divider()

    # 4. REPLAY MODE (Section 19)
    st.markdown("### ⏪ Replay Mode & Step Scrubber")
    st.caption("Inspect receiver alignment, detector response, and decision telemetry at any historical timestep.")

    total_steps = len(results)
    if total_steps > 0:
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1, 1, 3])
        with c_ctrl1:
            step_idx = st.slider("Step Scrubber", min_value=0, max_value=total_steps - 1, value=total_steps - 1, step=1)
        with c_ctrl2:
            st.markdown(f"**Selected Step**: `{step_idx}` / `{total_steps - 1}`")

        sel_res = results[step_idx]
        sel_t = sel_res.get("timestamp", step_idx)
        sel_rx = sel_res.get("receiver", {})
        sel_rx_f = sel_rx.get("center_frequency_hz", 0.0) / 1e6
        sel_rx_bw = sel_rx.get("instantaneous_bandwidth_hz", 20e6) / 1e6
        sel_det = sel_res.get("detection", {})
        sel_out = sel_res.get("outcome", {})
        sel_sched = sel_res.get("scheduler", {})
        sel_gt = sel_res.get("ground_truth", {})
        sel_emitters = [e for e in sel_gt.get("emitters", []) if e.get("transmitting", False)]

        r1, r2, r3, r4, r5, r6 = st.columns(6)
        with r1:
            tx_str = ", ".join([f"{e['frequency_hz']/1e6:.1f} MHz" for e in sel_emitters]) if sel_emitters else "Silent"
            st.metric("Emitter Freq [Truth]", tx_str)
        with r2:
            st.metric("Receiver Scanned", f"{sel_rx_f:.1f} MHz")
        with r3:
            st.metric("Receiver BW", f"{sel_rx_bw:.1f} MHz")
        with r4:
            st.metric("Detected?", "✅ YES" if sel_det.get("detected") else "❌ NO")
        with r5:
            st.metric("Step Reward", f"{sel_out.get('reward', 0.0):+.2f}")
        with r6:
            st.metric("Step Outcome", sel_out.get("outcome", "NONE"))

        st.caption(f"**Diagnostic at Step {sel_t}**: {sel_res.get('diagnostic_reason', '')}")

    st.divider()

    # 5. OPPORTUNITY INSPECTOR (Section 20)
    st.markdown("### 🔬 Opportunity Inspector")
    st.caption("Inspect individual transmission hops or burst opportunities to verify coverage, detection, and intercept delay.")

    opps_summary = env.get_opportunities_summary()
    if opps_summary:
        opp_df = pd.DataFrame(opps_summary)
        opp_ids = [o["Opportunity ID"] for o in opps_summary]

        col_sel, col_det = st.columns([1, 2])
        with col_sel:
            selected_opp_id = st.selectbox("Select Opportunity to Inspect", opp_ids, index=0)
            chosen_opp = next((o for o in opps_summary if o["Opportunity ID"] == selected_opp_id), None)

        with col_det:
            if chosen_opp:
                st.markdown(f"#### Details for `{selected_opp_id}`")
                q1, q2, q3, q4 = st.columns(4)
                with q1:
                    st.metric("Emitter ID", chosen_opp.get("Emitter ID", "—"))
                    st.metric("Scope", chosen_opp.get("Scope", "HOP"))
                with q2:
                    st.metric("Start Step", chosen_opp.get("Start Step", "—"))
                    st.metric("End Step", chosen_opp.get("End Step", "—"))
                with q3:
                    st.metric("Frequency", chosen_opp.get("Frequency", "—"))
                    st.metric("Duration", f"{chosen_opp.get('Duration', '—')} steps")
                with q4:
                    st.metric("Receiver Covered?", chosen_opp.get("Receiver Entered", "—"))
                    st.metric("Intercept Status", chosen_opp.get("Detected", "—"))

                st.caption(f"First Intercept: **{chosen_opp.get('First Intercept', '—')}** | Delay: **{chosen_opp.get('Delay (Steps)', '—')} steps** | Final Status: **{chosen_opp.get('Status', '—')}**")

        with st.expander("📋 View Complete Opportunities Table"):
            st.dataframe(opp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No transmission opportunities recorded.")
