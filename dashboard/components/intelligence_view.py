from __future__ import annotations

from typing import Any
import altair as alt
import streamlit as st
import pandas as pd


def render_intelligence_view(sim_data: dict[str, Any]) -> None:
    """Renders Tab 4: Algorithm Intelligence

    Features:
    - Learned Transition Matrix heatmap P(f_{next} | f_{prev})
    - Current prediction panel
    - Algorithm-specific explanation panel (UCB1, Thompson, SW-UCB, Discounted TS, Context-Aware)
    - Periodic aliasing diagnostic panel
    - Prediction log
    """
    env = sim_data["env"]
    trans_matrix_df = sim_data.get("transition_matrix_df")
    preds_df = sim_data.get("predictions_df")
    arm_df = sim_data.get("arm_df")
    scheduler_name = sim_data["scheduler_name"]
    results = sim_data.get("results", [])
    last_res = results[-1] if results else {}
    sched_state = env.scheduler.get_state()
    expl = sched_state.explanation or {}
    scenario = sim_data.get("scenario", {})

    st.subheader(f"🧠 Algorithm Intelligence & Internal State: {scheduler_name.upper()}")
    st.markdown("""
    Inspects internal algorithmic models, statistical representations, and learned transition dynamics.
    All displayed values reflect authentic internal scheduler state.
    """)

    # 1. Transition Model & Predictions
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔄 Observed Transition Matrix")
        st.caption(r"$P(f_{\text{next}} \mid f_{\text{prev}})$ learned strictly from observed receiver detections.")
        if trans_matrix_df is not None and not trans_matrix_df.empty:
            matrix_reset = trans_matrix_df.reset_index().rename(columns={"index": "Source Band (MHz)"})
            melted = matrix_reset.melt(id_vars=["Source Band (MHz)"], var_name="Target Band (MHz)", value_name="Probability")
            heatmap = (
                alt.Chart(melted)
                .mark_rect()
                .encode(
                    x=alt.X("Target Band (MHz):N", title="Next Frequency (MHz)"),
                    y=alt.Y("Source Band (MHz):N", title="Previous Frequency (MHz)"),
                    color=alt.Color("Probability:Q", scale=alt.Scale(scheme="blues")),
                    tooltip=["Source Band (MHz)", "Target Band (MHz)", alt.Tooltip("Probability:Q", format=".2%")],
                )
                .properties(height=280)
            )
            st.altair_chart(heatmap, use_container_width=True)
        else:
            st.info("No transition matrix recorded.")

    with col2:
        st.markdown("### 🔮 Current Prediction & Accuracy")
        st.caption("Observed next-frequency anticipation state.")

        last_f = None
        if last_res.get("observation", {}).get("recent_history"):
            last_f = last_res["observation"]["recent_history"].get("last_detected_frequency_hz")

        pred_f = getattr(env.scheduler, "predicted_frequency_hz", None)
        prob_val = expl.get("transition_probability", 0.0)

        prev_str = f"{last_f/1e6:.1f} MHz" if last_f else "None"
        pred_str = f"{pred_f/1e6:.1f} MHz" if pred_f else (f"{expl.get('action_mhz', 0.0):.1f} MHz" if "action_mhz" in expl else "None")
        action_str = f"{sched_state.selected_frequency_hz/1e6:.1f} MHz" if sched_state.selected_frequency_hz else "None"

        pred_box = f"""
```text
Previous observed:
{prev_str}

Predicted next:
{pred_str}

Probability / Confidence:
{prob_val:.1%}

Selected action:
{action_str}
```
"""
        st.markdown(pred_box)

        if preds_df is not None and not preds_df.empty:
            correct_count = (preds_df["Outcome"] == "✅ Correct").sum()
            total_count = len(preds_df)
            acc = correct_count / total_count if total_count else 0.0
            st.metric("Observed Transition Prediction Accuracy", f"{acc:.2%}", f"{correct_count}/{total_count} valid transitions")

    st.divider()

    # 2. Algorithm-Specific Explanations (Section 15)
    st.markdown("### 📐 Algorithm-Specific Parameter Explanations")

    if scheduler_name == "context_aware":
        st.markdown("""
        **Context-Aware 3-Term Score Formula**:  
        $\\text{Score}(j) = w_{\\text{trans}} \\times P(f_j \\mid f_{\\text{prev}}) + w_{\\text{act}} \\times \\text{Activity}(j) + w_{\\text{exp}} \\times \\text{Exploration}(j)$
        """)
        c_a, c_b, c_c, c_d = st.columns(4)
        with c_a:
            st.metric("Transition Probability", f"{expl.get('transition_probability', 0.0):.3f}")
        with c_b:
            st.metric("Recency / Activity Score", f"{expl.get('activity_level', 0.0):.3f}")
        with c_c:
            st.metric("Exploration Bonus", f"{expl.get('exploration_bonus', 0.0):.3f}")
        with c_d:
            st.metric("Final Action Score", f"{expl.get('estimated_value', 0.0):.3f}")

    elif scheduler_name == "ucb1":
        st.markdown("""
        **UCB1 Formula**:  
        $\\text{UCB}_i = \\hat{\\mu}_i + c \\sqrt{\\frac{\\ln t}{N_i(t)}}$
        """)
        u1, u2, u3, u4 = st.columns(4)
        with u1:
            st.metric("Selected Band", f"{expl.get('action_mhz', 0.0):.1f} MHz")
        with u2:
            st.metric("Estimated Reward", f"{expl.get('estimated_value', 0.0):.3f}")
        with u3:
            st.metric("Exploration Bonus", f"{expl.get('exploration_bonus', 0.0):.3f}")
        with u4:
            ucb_score = (expl.get("estimated_value", 0.0) or 0.0) + (expl.get("exploration_bonus", 0.0) or 0.0)
            st.metric("UCB Score", f"{ucb_score:.3f}")

    elif scheduler_name == "thompson":
        st.markdown("""
        **Thompson Sampling**:  
        Samples $\\theta_i \\sim \\text{Beta}(\\alpha_i, \\beta_i)$ for each candidate frequency band.
        """)
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.metric("Selected Band", f"{expl.get('action_mhz', 0.0):.1f} MHz")
        with t2:
            st.metric("Beta Alpha (Success)", f"{expl.get('alpha', 1.0):.1f}")
        with t3:
            st.metric("Beta Beta (Failure)", f"{expl.get('beta', 1.0):.1f}")
        with t4:
            st.metric("Sampled Value", f"{expl.get('estimated_value', 0.0):.3f}")

    elif scheduler_name == "sw_ucb":
        st.markdown("""
        **Sliding Window UCB (SW-UCB)**:  
        Limits memory to the most recent $W$ observations: $\\hat{\\mu}_i(W) + c \\sqrt{\\frac{2 \\ln(\\min(t, W))}{N_i(W)}}$.
        """)
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.metric("Selected Band", f"{expl.get('action_mhz', 0.0):.1f} MHz")
        with w2:
            st.metric("Recent Window Mean", f"{expl.get('estimated_value', 0.0):.3f}")
        with w3:
            st.metric("Exploration Bonus", f"{expl.get('exploration_bonus', 0.0):.3f}")
        with w4:
            st.metric("Window Pull Count", f"{expl.get('window_pulls', 0)}")

    elif scheduler_name == "discounted_thompson":
        st.markdown("""
        **Discounted Thompson Sampling (D-TS)**:  
        Applies exponential decay factor $\\gamma < 1.0$ to past counts: $\\alpha_i \\leftarrow 1 + \\gamma(\\alpha_i - 1)$.
        """)
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric("Decay Factor (Gamma)", f"{expl.get('gamma', 0.95):.2f}")
        with g2:
            st.metric("Discounted Alpha", f"{expl.get('alpha', 1.0):.2f}")
        with g3:
            st.metric("Discounted Beta", f"{expl.get('beta', 1.0):.2f}")
        with g4:
            st.metric("Posterior Mean", f"{expl.get('estimated_value', 0.0):.3f}")

    else:
        st.info(f"Scheduler '{scheduler_name}': {expl.get('reason', 'Explanation unavailable.')}")

    st.divider()

    # 3. Periodic Aliasing Diagnostic (Section 12)
    st.markdown("### 🛰️ Periodic Aliasing & Stroboscopic Blindness Diagnostic")
    emitters = scenario.get("emitters", [])
    has_periodic = any(e.get("time_behavior", {}).get("type") == "periodic" for e in emitters)

    if has_periodic:
        periodic_emitter = next(e for e in emitters if e.get("time_behavior", {}).get("type") == "periodic")
        tb = periodic_emitter.get("time_behavior", {})
        on_d = tb.get("on_duration", 10)
        off_d = tb.get("off_duration", 10)
        emitter_period = on_d + off_d
        scan_cycle = len(env.scheduler.bands_hz)

        st.markdown(f"""
        - **Emitter Period**: `{emitter_period}` steps (`{on_d}` ON, `{off_d}` OFF)
        - **Receiver Scan Cycle**: `{scan_cycle}` bands
        - **Scan / Emitter Period Ratio**: `{scan_cycle / max(emitter_period, 1):.2f}`
        """)

        if scan_cycle % emitter_period == 0 or emitter_period % scan_cycle == 0:
            st.warning(f"""
            ⚠️ **Potential Periodic Aliasing Detected!**  
            The receiver scan cycle length ({scan_cycle} steps) is harmonically synchronized with the emitter period ({emitter_period} steps).
            If the initial phase offset does not align, the receiver will experience **persistent stroboscopic blindness** (0% intercept ratio)
            despite scanning the correct channel at regular intervals.
            """)
        else:
            st.success(f"""
            ✅ **Coprime / Non-Harmonic Scan Schedule**:  
            Scan cycle length ({scan_cycle}) and emitter period ({emitter_period}) are non-synchronized.
            Relative phase precession guarantees that the receiver will periodically overlap active transmission bursts.
            """)
    else:
        st.info("No periodic emitters in this scenario. Stroboscopic aliasing check is inactive.")
