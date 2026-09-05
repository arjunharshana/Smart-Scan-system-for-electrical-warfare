# 📡 RF Environment & Cognitive Algorithm Intelligence Dashboard

A modern, interactive simulation and telemetry dashboard built with **Streamlit** to evaluate, test, and benchmark cognitive radio scanning algorithms across dynamic RF environments.

---

## 🚀 Quickstart

1. **Activate Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Install Dependencies** (if not already installed):
   ```bash
   pip install streamlit pandas altair pyyaml
   ```

3. **Launch Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
   *(Or alternatively run `streamlit run dashboard.py` from the root)*

---

## 🎛️ Features & Capabilities

### 1. 📊 KPI & Statistical Detection Analytics
- Executive summary metrics: Total Scans, Hits, Misses, False Alarms, Correct Rejections.
- Detection performance metrics: Probability of Detection ($P_d$), False Alarm Rate ($P_{fa}$), Interception Ratio, Prediction Accuracy, and Average Reward.
- Detailed Emitter discovery status with Time-to-First-Intercept latency.

### 2. 🌊 Spectral Waterfall Spectrogram
- Interactive 2D waterfall plot mapping time vs frequency for active emitters and receiver scanning windows.
- Receiver hardware profile (Instantaneous Bandwidth, Sensitivity, Noise Floor, Detection Threshold).

### 3. 📈 Algorithm Dynamics & Convergence
- Cumulative Reward curves demonstrating search efficiency.
- 20-step rolling hit rate illustrating learning speed and convergence.
- Step-by-step instantaneous reward and receiver tuning trajectories.

### 4. 🎰 Bandit Arm Dynamics (UCB1 & Thompson Sampling)
- Multi-Armed Bandit visitation counts across frequency channels.
- Empirical mean reward and posterior parameter breakdown per band.
- Quantified exploration vs. exploitation metrics.

### 5. ⚔️ Multi-Algorithm Benchmark Comparison
- One-click comparative execution testing **Sequential**, **Random**, **UCB1**, and **Thompson Sampling** on the exact same scenario and seed.
- Multi-algorithm Cumulative Reward and Rolling Hit Rate comparison charts.
- Summary scoreboard table identifying the optimal scanning algorithm.

### 6. 🛠️ Scenario & Hardware Sandbox
- Interactive GUI builder to define custom Radar and Communication emitters with hopping/sweeping/burst patterns.
- Configurable receiver bandwidth, noise floor, sensitivity, and spectrum frequency ranges.
