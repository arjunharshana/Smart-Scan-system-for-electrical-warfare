# Literature Evidence Table: Theoretical Verification

**Project**: SIH26055 / DRDO Smart Scan Strategy for Electronic Warfare  
**Status**: Focused Peer-Reviewed Literature Verification Audit  
**Date**: September 2026  

---

## 1. Authoritative Evidence Matrix

| Paper / Authors | Problem Studied | Model Formulated | Important Mathematical Assumptions | Relevant to Simulator? | Scientific Evidence & Implications |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **Zhao, Tong, Swami, & Chen (2007)**<br>*IEEE JSAC* | Decentralized cognitive MAC for opportunistic spectrum access. | **Partially Observable Markov Decision Process (POMDP)**. | 1. Channel occupancy follows Markov transitions.<br>2. Secondary user senses $k < N$ channels.<br>3. Belief state $\mathbf{b}_t$ is updated via Bayes' rule. | **YES (Direct)** | Proves that recursive Bayesian belief vector $\mathbf{b}_t \in \Delta^{N-1}$ is a **sufficient information state** for optimal channel selection under partial observation censoring. |
| **Liu & Zhao (2010)**<br>*IEEE Trans. Information Theory* | Dynamic multi-channel opportunistic access under resource constraints. | **Restless Multi-Armed Bandit (RMAB)** with Whittle index. | 1. **Channels are mutually independent** ($P(S_{t+1} \mid S_t) = \prod P_i(s_{t+1}^{(i)} \mid s_t^{(i)})$).<br>2. Two-state Gilbert-Elliott Markov arms.<br>3. Verified indexability condition. | **PARTIALLY (Caution)** | Proves Whittle indexability for **independent channels**. In our simulator, the single hopping emitter creates **correlated channels**, violating the independence assumption. Confirms our scheduler is a heuristic priority score, not a provably optimal Whittle policy. |
| **Yu (2010)**<br>*Artificial Intelligence* | Modeling hidden processes with non-geometric holding times. | **Hidden Semi-Markov Model (HSMM)**. | 1. Transitions occur at discrete jump epochs.<br>2. State holding time follows explicit duration distribution $p_i(d)$.<br>3. Memoryless only at jump epochs. | **YES (Exact)** | Proves standard HMM geometric dwell ($P(d) = (1-p)p^{d-1}$) creates severe structural errors for physical processes with fixed/minimum dwell. Validates that dwell duration must be explicitly parameterized. |
| **Wang, Wang, & Adachi (2012)**<br>*IEEE Trans. Wireless Communications* | Primary user traffic classification in cognitive radio. | **Hidden Semi-Markov Model (HSMM)** for packet dwell. | 1. Primary user packet transmissions follow fixed or bounded holding times.<br>2. Secondary sensing is discrete-time. | **YES (Direct)** | Demonstrates that HSMM achieves $35\%-50\%$ higher prediction accuracy than HMM on spectrum traffic because physical transmissions have explicit packet holding times. |
| **Hausknecht & Stone (2015)**<br>*AAAI Conference on AI* | Reinforcement learning in partially observable environments. | **Deep Recurrent Q-Network (DRQN / LSTM-DQN)**. | 1. Recurrent hidden state $h_t$ acts as a learned surrogate belief state.<br>2. Training transitions sampled as contiguous sequential traces. | **YES (Direct)** | Establishes that an LSTM can approximate belief tracking without an analytical state estimator, but notes slow sample efficiency and reliance on extensive offline training. |
| **French (1999)**<br>*Trends in Cognitive Sciences* | Catastrophic forgetting and interference in connectionist networks. | **Multi-Layer Recurrent & Feed-Forward Neural Networks**. | 1. Shared synaptic weight matrices.<br>2. Sequentially presented non-stationary tasks or diverse input distributions. | **YES (Direct)** | Explains why V4.1 LSTM-DDQN suffered performance collapse ($80\% \to 26.9\%$) when trained on 6 mixed regimes: opposing gradient updates ($\nabla \mathcal{L}_A \cdot \nabla \mathcal{L}_B < 0$) across disparate cycle periods overwrite shared recurrent weights. |
| **Torrieri (2018)**<br>*Springer (4th Edition)* | Electronic warfare and spread spectrum intercept receivers. | **Slow Frequency Hopping (SFH) Signal Interception**. | 1. Transmitters dwell for $D \ge 1$ coherent intervals.<br>2. Intercept receiver bandwidth $B_{\text{rx}} < B_{\text{spread}}$.<br>3. Receiver must sweep/step across discrete bands. | **YES (Domain Exact)** | Confirms that physical frequency-hopping radar and communications operate with discrete dwell times and fixed frequency tables, matching our simulator architecture. |

---

## 2. Synthesis of Mathematical Preconditions

```
[Independent Channels: P(S'|S) = ∏ P_i] ──(Violated by Single Hopping Emitter)──► Formal RMAB / Whittle Index
                                                                                           │
[Partial Channel Censoring: B_rx < B_spread] ──────────────────────────────────────────────┼──► Augmented Belief-POMDP
                                                                                           │      (EXACT MATCH)
[Explicit Dwell Duration: dwell_steps = D] ────(Violated by 1st-Order HMM/CA)─────────────┼──► Augmented State (F_t, τ_t)
                                                                                           │      (EXACT MATCH)
[Diverse Multi-Regime Mixture: T ∈ {6,8,9,12,15}] ──(Causes Gradient Conflict)───────────┴──► Monolithic LSTM-DDQN
                                                                                                  (VULNERABLE)
```
