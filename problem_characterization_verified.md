# Problem Characterization Verification Audit

**Project**: SIH26055 / DRDO Smart Scan Strategy for Electronic Warfare  
**Status**: Critical System Identification & Code-Level Verification Pass  
**Verification Standard**: Evidence from Code + Authoritative Literature; No Unsubstantiated Hypotheses  
**Date**: September 2026  

---

## 1. Audit Table of Previous Claims

Every major claim from previous reports is classified: **VERIFIED**, **PARTIALLY VERIFIED**, **UNSUPPORTED**, or **FALSE**.

| # | Claim | Evidence from Code | Literature Evidence | Status | Correction / Nuance |
| :- | :--- | :--- | :--- | :---: | :--- |
| 1 | **"Frequency-only state is non-Markov"** | [`frequency_behaviors/hopping.py:45`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/frequency_behaviors/hopping.py#L45): `hop_index = time_step // dwell_steps`. For $D > 1$, $P(F_{t+1}=F_t \mid F_t)$ is $1$ if elapsed dwell $\tau < D-1$, but $0$ if $\tau = D-1$. | Rabiner (1989), Yu (2010): processes with deterministic/non-geometric holding times violate memorylessness in the state variable alone. | **VERIFIED** | Frequency alone is non-Markovian; however, augmenting frequency with elapsed dwell counter $(F_t, \tau_t)$ creates an **exact discrete-time Markov chain**. |
| 2 | **"The process is semi-Markov"** | [`frequency_behaviors/hopping.py:24-54`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/frequency_behaviors/hopping.py#L24-L54): Transitions between distinct frequency states occur after an integer holding time `dwell_steps`. | Cinlar (1975), Howard (1971): A stochastic process where transitions at jump times form a Markov chain and holding time depends on state is a semi-Markov process. | **VERIFIED** | Specifically, it is a **discrete-time semi-Markov process with degenerate (deterministic constant) dwell distributions**. |
| 3 | **"HSMM is the exact mathematical model"** | [`frequency_behaviors/hopping.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/frequency_behaviors/hopping.py): State stays for fixed $D$ steps. [`receiver/detector.py:35-55`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/receiver/detector.py#L35-L55): Observations are emitted conditionally on true state. | Yu (2010), Wang et al. (2012): An HSMM with deterministic duration $p_i(d) = \delta(d - D)$ represents fixed-dwell hidden states. | **PARTIALLY VERIFIED** | **Nuance**: An HSMM represents the *passive emitter dynamics* under observation noise. However, our problem includes **action-dependent sensing** (tuner selection $A_t$). The complete decision problem is a **Controlled HSMM (or Augmented Belief-POMDP)**, not a passive HSMM. |
| 4 | **"POMDP is appropriate"** | [`environment/rf_environment.py:263-325`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/environment/rf_environment.py#L263-L325): Receiver senses 1 of 30 bins; observations $Y_t$ depend on action $A_t$; true state $S_t$ is hidden. | Zhao & Sadler (2007), Zhao et al. (2007): Sequential spectrum channel selection under partial sensing is a canonical POMDP. | **VERIFIED** | It is formally an augmented POMDP where state $S_t = (F_t, \tau_t, \text{tx}_t)$ and belief $\mathbf{b}_t \in \Delta^{|\mathcal{S}|-1}$ is a sufficient information state. |
| 5 | **"Whittle / RMAB is incompatible"** | [`emitters/base.py:33-55`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/emitters/base.py#L33-L55): Single emitter determines spectrum. Occupancy across all 30 bins is mutually exclusive ($\sum \mathbf{1}_{F_t = f_i} \le 1$). | Whittle (1988), Liu & Zhao (2010): Exact Whittle indexability requires arm transitions to be mutually independent. | **PARTIALLY VERIFIED** | **Correction**: Whittle is incompatible as a **formally guaranteed optimal index policy** because arms are coupled. However, decoupled priority indices remain a valid **heuristic ranking algorithm** (as proven by our Whittle-style W3 baseline). |
| 6 | **"LSTM is poorly matched / overparameterized"** | In capacity ablation, $H=64$ achieves $44.2\%$ test IR while $H=256$ collapses to $21.4\%$; multi-regime training collapses test IR to $26.9\%$. | French (1999), Lee et al. (2019): Shared recurrent weights suffer catastrophic gradient interference across disparate temporal periods. | **PARTIALLY VERIFIED** | **Correction**: LSTM is **not theoretically incapable**; an LSTM with sufficient data can approximate belief filters. The limitation is **sample efficiency and inductive bias**: gradient descent across conflicting temporal regimes destroys shared recurrent weights. |
| 7 | **"Current benchmark mainly tests memorization"** | [`scenarios_v4_1.py`](file:///home/rishant-gupta/Projects/SIH-2026/benchmarks/scenarios_v4_1.py): 7 of 8 scenarios use 3 bins in a static cyclic loop repeating 20–100 times per episode. Zero within-episode regime shifts. | - | **VERIFIED** | 27 of 30 channels are dead space. A scheduler locking onto a static 9-step cycle achieves high IR without performing general cognitive inference. |
| 8 | **"Single emitter creates cross-channel coupling"** | [`environment/rf_environment.py:268-274`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/environment/rf_environment.py#L268-L274): Only 1 emitter state exists; detection in bin $j$ implies absence in bin $k \ne j$. | Anandkumar et al. (2011): Mutual exclusivity induces strict negative correlation across unobserved channels. | **VERIFIED** | Detecting bin $j$ collapses belief across all $k \ne j$ to zero (modulo false alarms). |
| 9 | **"Dwell is deterministic / stochastic"** | [`frequency_behaviors/hopping.py:45`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/frequency_behaviors/hopping.py#L45): `hop_index = time_step // self.dwell_steps`. | - | **VERIFIED** | Ground-truth dwell is **strictly deterministic constant integer** in all current scenarios. There is zero stochastic dwell in the codebase. |
| 10 | **"Detector observations are sufficient"** | [`receiver/detector.py:35-55`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/receiver/detector.py#L35-L55): $Y_t \sim \text{Bernoulli}(P_D I_t + P_{FA}(1-I_t))$ with $P_D=0.95, P_{FA}=0.02$. | Zhao et al. (2007): Binary detector output with known $P_D, P_{FA}$ is fully sufficient for Bayesian belief recursion. | **VERIFIED** | Binary detections plus RSSI provide complete sufficient statistics for likelihood updates. |

---

## 2. Actual Code-Level Generative Model

### 2.1 Hidden State ($S_t$)
From [`rf_environment/emitters/base.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/emitters/base.py) and [`rf_environment/frequency_behaviors/hopping.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/frequency_behaviors/hopping.py):
The physical simulation state at step $t \in \mathbb{N}$ consists strictly of:
$$S_t = \left( F_t, \tau_t, \text{tx}_t, k_t, \theta \right)$$
- $F_t \in \{f_0, \dots, f_{N-1}\}$: Active emitter frequency (one of 30 bins).
- $\tau_t = t \pmod D \in \{0, 1, \dots, D-1\}$: Elapsed dwell step within the current hop.
- $\text{tx}_t \in \{0, 1\}$: Emitter transmission gating state (`Continuous` $\implies 1$; `Burst` $\implies \mathbf{1}_{(t - t_0) \pmod I < B}$).
- $k_t = (t // D) \pmod K \in \{0, \dots, K-1\}$: Hop index in the emitter's frequency sequence.
- $\theta$: Stationary scenario parameter tuple $(\mathcal{F}_{\text{sequence}}, D, \text{Mode}, I, B)$.

**Minimum State for Known Regime $\theta$**:
When the scenario parameters $\theta$ are fixed, the minimum sufficient hidden state is simply:
$$S_t = \left( F_t, \tau_t, \text{tx}_t \right)$$
For $N=30$ bins and maximum dwell $D=5$, the state space has size $|\mathcal{S}| \le 30 \times 5 \times 2 = 300$ states. For canonical 3-bin scenarios ($K=3, D=3$), $|\mathcal{S}| = 3 \times 3 \times 1 = 9$ states.

### 2.2 Action Space ($A_t$)
From [`rf_environment/receiver/receiver.py:31`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/receiver/receiver.py#L31) and [`rf_environment/domain/action.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/domain/action.py):
$$A_t \in \{0, 1, \dots, N-1\}, \quad N = 30$$
The scheduler exclusively controls the instantaneous center frequency of the receiver: $f_{\text{rx}}(t) = f_{\min} + \frac{\text{BW}}{2} + A_t \times \text{BW}$. The scheduler has no control over bandwidth, sensitivity, or dwell duration.

### 2.3 State Transition Kernel ($P(S_{t+1} \mid S_t, A_t)$)
From [`rf_environment/environment/rf_environment.py:238-245`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/environment/rf_environment.py#L238-L245):
Scanning is passive: emitter dynamics are completely independent of receiver action $A_t$:
$$P(S_{t+1} \mid S_t, A_t) = P(S_{t+1} \mid S_t)$$
- **Elapsed Dwell**: $\tau_{t+1} = (\tau_t + 1) \pmod D$ (deterministic).
- **Frequency**:
  $$F_{t+1} = \begin{cases}
  F_t & \text{if } \tau_t < D - 1 \\
  \mathcal{F}_{\text{sequence}}[(k_t + 1) \pmod K] & \text{if } \tau_t = D - 1 \text{ (Sequential)} \\
  \sim \text{Uniform}(\mathcal{F}_{\text{sequence}}) & \text{if } \tau_t = D - 1 \text{ (Random)}
  \end{cases}$$
- **Burst Gating**: $\text{tx}_{t+1} = \mathbf{1}_{\{(t + 1 - t_0) \pmod I < B\}}$ (deterministic).

### 2.4 Observation Emission ($P(O_t \mid S_t, A_t)$)
From [`rf_environment/receiver/detector.py:27-56`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/receiver/detector.py#L27-L56):
The observation $O_t$ received by the scheduler is the tuple:
$$O_t = \left( A_t, Y_t, \text{RSSI}_t, t, \mathbf{h}_{\text{recent}}, \boldsymbol{\tau}^{\text{scan}}, \tau^{\text{detect}} \right)$$
Where the stochastic observation variable is binary detection $Y_t \in \{0, 1\}$:
Let $I_t = \mathbf{1}_{\{F_t \in \text{bin}(A_t)\}} \cdot \text{tx}_t$ be the in-band active transmission indicator.
$$P(Y_t = 1 \mid S_t, A_t) = \begin{cases}
P_D = 0.95 & \text{if } I_t = 1 \\
P_{FA} = 0.02 & \text{if } I_t = 0
\end{cases}$$
Signal strength $\text{RSSI}_t$ is emitted only when $Y_t = 1$. Timers $\boldsymbol{\tau}^{\text{scan}}$ and $\tau^{\text{detect}}$ are deterministic functions of past actions and detections.

### 2.5 Reward Function ($R_t$)
From [`rf_environment/rewards/r4_reward.py:64-85`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/rewards/r4_reward.py#L64-L85):
$$R_t = \begin{cases}
1.00 + 0.15 \cdot \min\left(\max\left(\frac{\text{RSSI}_t - (-90)}{40}, 0\right), 1\right) & \text{if } Y_t = 1 \\
-0.05 & \text{if } Y_t = 0
\end{cases}$$
**CRITICAL VERIFIED FACT**: The reward $R_t$ depends **STRICTLY on detector output $Y_t$ and action $A_t$**. It does **NOT** access ground truth. A false alarm ($Y_t = 1$ in an empty bin) yields $R_t \ge 1.00$; a detector miss ($Y_t = 0$ in the correct bin) yields $R_t = -0.05$. The objective of RL is maximizing expected detector hits under noise.
