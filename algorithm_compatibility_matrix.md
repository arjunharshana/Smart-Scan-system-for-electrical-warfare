# Algorithm Compatibility Matrix & Theoretical Assumption Scorecard

**Project**: SIH26055 / DRDO Smart Scan Strategy  
**Scope**: Rigorous Theoretical Compatibility Analysis of Candidate Algorithmic Families  
**Target Problem**: Partially Observable Discrete-Time Hidden Semi-Markov Dynamic Spectrum Surveillance  
**Date**: September 2026  

---

## 1. Master Assumption Match Scorecard

Notation:
- **✓** = Naturally supported by foundational theory
- **△** = Supported with explicit engineering extensions / approximations
- **✗** = Incompatible; fundamental assumptions directly violated
- **?** = Theoretical evidence insufficient or non-constructive

| Algorithmic Family | Partial Observability | Temporal Dependence | Variable / Fixed Dwell | Cross-Channel Coupling | Regime Non-Stationarity | Sparse / Censored Data | Online Sample Efficiency | Engineering Interpretability | Overall Mathematical Fit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Context-Aware (CA)** | △ | △ | ✗ | ✓ | △ | ✓ | ✓ | ✓ | **Moderate (Heuristic)** |
| **Classical MAB (UCB/TS)** | ✗ | ✗ | ✗ | ✗ | ✗ | △ | ✓ | ✓ | **Poor (Incompatible)** |
| **Restless Bandits (RMAB)** | △ | ✓ | ✗ | ✗ | ✗ | ✓ | △ | ✓ | **Poor (Violated)** |
| **Whittle Index Policy** | △ | ✓ | ✗ | ✗ | ✗ | ✓ | △ | ✓ | **Poor (Violated)** |
| **Hidden Markov Model (HMM)** | ✓ | ✓ | ✗ | ✓ | △ | ✓ | ✓ | ✓ | **Moderate (Dwell Gap)** |
| **Hidden Semi-Markov (HSMM)**| **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **EXACT / IDEAL FIT** |
| **POMDP (Belief-State)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | △ | **✓** | **EXACT / FORMAL FIT** |
| **Feed-Forward DDQN (V4.0)** | ✗ | △ | ✗ | △ | ✗ | ✗ | ✗ | ✗ | **Poor (Overfits)** |
| **Recurrent RL / DRQN (V4.1)** | ✓ | ✓ | △ | △ | △ | △ | ✗ | ✗ | **Moderate (Empirical)** |
| **Transformer / Decision Trans.**| ✓ | ✓ | ✓ | △ | △ | ✗ | ✗ | ✗ | **Poor (Data Inefficient)**|
| **Model-Based RL (MBRL)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **STRONG / HIGH FIT** |

---

## 2. Deep-Dive Assumption Breakdown by Algorithm Family

### 2.1 Context-Aware Scheduler (CA)
- **Foundational Assumptions**:
  1. 1st-order stationary Markov transitions: $P(F_{t+1} \mid F_t)$.
  2. Memoryless dwell: probability of remaining on a channel is geometrically distributed.
  3. Single-step empirical counting suffices for sequence extraction.
- **What It Models Well**:
  - Direct channel transition correlation: quickly extracts $\hat{T}(j \to i)$ from confirmed detections.
  - Coupled channel activity: updates the global transition matrix across all bins simultaneously.
- **What It Cannot Model**:
  - Dwell time progression: cannot distinguish whether the emitter has stayed on channel for 1 step or is at the end of its dwell.
  - Multi-hop temporal memory: cannot distinguish paths where the next hop depends on a previous sequence context ($F_{t-1}$).
- **Compatibility Verdict**: **Moderate / Baseline**. Excellent for quick online transition estimation, but fails when dwell duration is non-trivial ($D > 1$).

---

### 2.2 Multi-Armed Bandits (MAB - UCB1, Thompson Sampling)
- **Foundational Assumptions**:
  1. Arms are stationary: reward distributions $R_i \sim \mathcal{D}_i$ do not evolve over time.
  2. Passive arms remain frozen: state does not change when unobserved.
  3. Arm independence: pulls on arm $i$ convey zero information about arm $j$.
- **Why It Fails Completely**:
  - Emitters hop dynamically: an arm that was transmitting at step $t$ is guaranteed to be vacant at $t+3$.
  - MAB averages rewards over all past time steps, completely destroying temporal structure.
- **Compatibility Verdict**: **Fundamentally Incompatible**. Must be eliminated from consideration.

---

### 2.3 Restless Bandits & Whittle Index Policies (RMAB)
- **Foundational Assumptions** (Whittle 1988, Liu & Zhao 2010):
  1. **Arm Independence**: Arms evolve according to independent Markov chains:
     $$P(S_{t+1} \mid S_t) = \prod_{i=1}^N P_i(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)})$$
  2. **Decoupled State**: Each arm has an internal state $s^{(i)} \in \{0, 1\}$ that transitions independently.
  3. **Indexability**: The set of passive states under subsidy $W$ must increase monotonically from $\emptyset$ to the full state space as $W$ increases.
- **Why Our Problem Violates RMAB Foundations**:
  1. **Single Emitter Constraint**: A single frequency-hopping emitter creates strict informational and physical coupling. If channel $j$ is active, channel $k$ is idle.
  2. **Coordinated Transitions**: Transitions are correlated across channels. Observing a detection on arm $j$ alters the transition probabilities of all other arms.
  3. **Non-Separable Rewards**: The reward cannot be factored into independent per-arm value functions without severe distortion.
- **Compatibility Verdict**: **Mathematically Incompatible as a Formal RMAB**. Can only be operated as an **ad-hoc heuristic index policy** (as in our Whittle-style scheduler), not as an optimal index policy.

---

### 2.4 Hidden Markov Models (HMM)
- **Foundational Assumptions** (Rabiner 1989):
  1. The hidden state $X_t \in \{1, \dots, N\}$ satisfies the 1st-order Markov property: $P(X_{t+1} \mid X_t, \dots, X_0) = P(X_{t+1} \mid X_t)$.
  2. Observations $Y_t$ are conditionally independent given $X_t$.
  3. Sojourn / dwell times in state $X_t$ are geometrically distributed: $P(d) = (1 - a_{ii}) a_{ii}^{d-1}$.
- **What It Models Well**:
  - Direct partial observability: forward-backward belief propagation updates $b_i(t) = P(X_t = i \mid Y_{1:t})$ cleanly via Bayes' rule.
  - Detector error handling: naturally incorporates $P_D$ and $P_{FA}$ emission matrices.
- **What It Cannot Model**:
  - Non-geometric dwell times: in our simulator, dwell is fixed ($D = 3$ steps). An HMM forces a memoryless geometric decay, predicting the highest probability of leaving the state immediately after entering it ($d=1$), exactly opposite to reality.
- **Compatibility Verdict**: **Moderate**. High belief tracking capability, but handicapped by the geometric dwell assumption.

---

### 2.5 Hidden Semi-Markov Models (HSMM)
- **Foundational Assumptions** (Yu 2010, Wang et al. 2012):
  1. The hidden state sequence $X_1, X_2, \dots$ forms a semi-Markov chain where transitions occur at random or deterministic epochs.
  2. When entering state $i$, a dwell duration $d \sim p_i(d)$ is chosen from an explicit duration distribution.
  3. The system remains in state $i$ for duration $d$ before transitioning to state $j$ according to transition matrix $A_{ij}$ ($A_{ii} = 0$).
- **Why HSMM Is the Exact Mathematical Twin of Our Simulator**:
  1. **Matches Constant / Variable Dwell**: The dwell distribution $p_i(d)$ can be parameterized as a deterministic Kronecker delta $p_i(d) = \delta(d - D)$, a Gaussian, or an empirical histogram.
  2. **Explicit Elapsed Dwell Counter**: The joint state $(X_t, \tau_t)$ tracks both the frequency and how many steps the emitter has remained in it.
  3. **Exact Posterior Inference**: Enables exact filtering of $P(X_{t+1} = j \mid Y_{1:t}, A_{1:t})$ using extended forward algorithms.
- **Compatibility Verdict**: **PERFECT THEORETICAL MATCH**. Represents the native generative structure of the RF simulator.

---

### 2.6 Partially Observable Markov Decision Processes (POMDP)
- **Foundational Assumptions** (Kaelbling et al. 1998, Zhao et al. 2007):
  1. Underlying environment is an MDP over state space $\mathcal{S}$.
  2. Observations $O \in \Omega$ are generated via emission distribution $O(o \mid s, a)$.
  3. Agent maintains an exact belief state $b(s) = P(S_t = s \mid H_t) \in \Delta^{|\mathcal{S}|-1}$.
- **Why POMDP Is the Correct Control Formulation**:
  - By augmenting the state to include elapsed dwell: $S_t = (F_t, \tau_t, \text{tx}_t)$, the semi-Markov process is converted into an equivalent augmented Markov chain.
  - The belief update $b_{t+1} = \text{BayesFilter}(b_t, a_t, y_t)$ is a sufficient statistic for optimal decision making.
- **Compatibility Verdict**: **EXACT CONTROL FORMULATION**. The theoretical gold standard for cognitive scan scheduling.

---

### 2.7 Feed-Forward Double DQN (V4.0)
- **Foundational Assumptions**:
  1. Observation input vector $x_t$ is fully informative of environment state (Markov property).
  2. Experience replay samples transitions uniformly at random ($s, a, r, s'$), assuming i.i.d. samples.
- **Why It Degrades in Dynamic Spectrum Access**:
  - The environment is partially observable. A static feature vector derived from a sliding window cannot represent continuous belief updates.
  - Uncorrelated replay sampling breaks temporal coherence required to learn sequential phase shifts.
  - Strong tendency to memorize specific frequency indices (e.g. $[5, 15, 25]$), collapsing on unseen permutations.
- **Compatibility Verdict**: **Poor**. Prone to severe pattern memorization and lack of generalization.

---

### 2.8 Recurrent RL / LSTM-DDQN (V4.1)
- **Foundational Assumptions** (Hausknecht & Stone 2015):
  1. The recurrent hidden state $h_t$ acts as a learned, continuous sufficient statistic representing the history $H_t$.
  2. Backpropagation through time (BPTT) allows gradients to flow across temporal sequences.
- **Strengths & Limitations in Our Problem**:
  - **Strength**: Can represent arbitrary temporal memory, elapsed dwell, and periodicities without manual feature engineering.
  - **Critical Failure Mechanism**: Parameter interference across diverse regimes. When trained on multiple regimes (dwell 2, 3, 4, 5, bursts), gradient updates from one regime overwrite the recurrent weights of another, causing catastrophic performance collapse ($80\% \to 26\%$).
- **Compatibility Verdict**: **Moderate / High Complexity**. Capable of learning single regimes, but fundamentally vulnerable to cross-regime parameter interference during offline Bellman updates.

---

## 3. Whittle-Specific 11-Point Verification Check

| Question | Mathematical Fact in Simulator | Scientific Conclusion |
| :--- | :--- | :--- |
| **1. What is an arm?** | A $20\text{ MHz}$ frequency bin $i \in \{0, \dots, 29\}$. | Well-defined spatial/spectral division. |
| **2. What is its state?** | Occupied ($1$) or Idle ($0$). | Binary state per channel. |
| **3. What happens when scanned?** | Generates detection $Y_t \sim \text{Bernoulli}(P_D I_t + P_{FA}(1-I_t))$. | State is partially revealed. |
| **4. What happens when NOT scanned?** | Arm state continues to evolve unobserved as emitter hops. | Restless condition holds. |
| **5. Does its state evolve while passive?** | **YES.** An unobserved channel can become active or idle. | Truly restless. |
| **6. Is transition independent of other arms?** | **NO.** The emitter hops directly from arm $j$ to arm $k$. | **VIOLATES RMAB ASSUMPTION**. |
| **7. Is reward separable across arms?** | **NO.** Total reward depends on capturing the single emitter. | **VIOLATES REWARD SEPARABILITY**. |
| **8. Can belief be maintained independently?** | **NO.** Bayes' update on arm $j$ changes the belief on arm $k$. | **VIOLATES BELIEF DECOUPLING**. |
| **9. Is single-arm subproblem well-defined?** | No. An arm's transition matrix depends on where the emitter was. | Cannot decouple Lagrangian subsidy. |
| **10. Is the model proven indexable?** | No proof exists for coupled multi-channel hopping emitters. | **INDEXABILITY UNPROVEN**. |
| **11. Can we legitimately call it Whittle?** | Only as a **"Whittle-Style Heuristic Index"**. | **MUST NOT CLAIM FORMAL WHITTLE**. |

---

## 4. LSTM Compatibility Check: Why Multi-Regime Fails

### 1. Does optimal action require long-term history?
**NO.** The true minimal sufficient statistic is $Z_t = (b_t, \tau_t, D, \hat{T})$. Once the active sequence and dwell are identified (requiring $\sim 10$ steps), only the immediate belief and elapsed dwell step are needed. An infinite recurrent memory is mathematically redundant.

### 2. Why does LSTM fail when multiple regimes are mixed?
- In single-regime training (Scenario 1), the LSTM learns fixed internal oscillator weights $W_{hh}$ synchronized to $T = 9$ steps ($3 \text{ hops} \times 3 \text{ dwell}$).
- When trained across 6 regimes with disparate periods ($T \in \{6, 8, 9, 12, 15\}$), the shared recurrent weights receive conflicting gradient updates:
  $$\nabla_\theta \mathcal{L}_{\text{Regime 1}} \cdot \nabla_\theta \mathcal{L}_{\text{Regime 2}} < 0$$
- This gradient conflict causes catastrophic representational collapse. Widening hidden capacity ($H=64 \to 128 \to 256$) merely provides more parameters to overfit the training mixture without solving gradient interference.

### 3. When IS an LSTM appropriate?
An LSTM is appropriate when:
1. Threat hopping dynamics follow high-dimensional, non-Markovian continuous functions (e.g. chaotic frequency modulation, complex stateful cryptoperiods).
2. The operational regime is stationary and uniform.

---

## 5. HMM vs HSMM: Detailed Dwell-Time Analysis

```
HMM (Geometric Dwell Decay):
P(stay = d) = (1 - p) * p^(d-1)
  │ █
  │ █ ▄
  │ █ ▄ ▂
  │ █ ▄ ▂  .
  └────────────► Duration d (Highest probability at d=1; strictly memoryless)

HSMM (Explicit Dwell Distribution):
P(stay = d) = Delta(d - D) or Gaussian(d; D, sigma^2)
  │       █
  │       █
  │       █
  │ ░ ░ ░ █ ░ ░
  └────────────► Duration d (Zero probability of leaving until d = D)
```

### Comparative Analysis
1. **Underlying Physics**: Real radar systems dwell on a frequency for a coherent processing interval (CPI) or pulse batch. The probability of hopping at $d < D$ is virtually zero.
2. **HMM Mismatch**: An HMM assigns maximum transition probability to the very first step ($d=1$), undercutting dwell persistence.
3. **HSMM Precision**: An HSMM explicitly models the dwell counter $\tau$, preserving high confidence during the dwell window and triggering a hop prediction precisely when $\tau = D - 1$.

---

## 6. Final Theoretical Recommendation: The Right Baseline Family

Based strictly on mathematical compatibility:

### 1. Primary Theoretical Baseline: Hidden Semi-Markov Model (HSMM) / Belief POMDP
- **Rationale**: Exactly matches the PO-HSMM generative structure of the simulator. Separates fast online parameter tracking (dwell and transitions) from state filtering.
- **Advantage**: Zero offline gradient training required; immune to cross-regime parameter interference; 100% interpretable Bayesian belief states.

### 2. Secondary Fast Baseline: Context-Aware with Adaptive Dwell Tracking (CA-Dwell)
- **Rationale**: Augments the current Context-Aware transition matrix with an explicit online dwell estimator $\hat{d}$, eliminating CA's memoryless blind spot.

### 3. Tactical Neural Benchmark: Frozen V4.1 LSTM-Hybrid ($H=64$)
- **Rationale**: Retain as the primary deep RL benchmark for SIH presentation, recognizing its strengths on known structured threats and its explicit operational boundaries.

### 4. Families to Formally Discontinue:
- **Classical MAB (UCB1, Thompson Sampling)**: Formally incompatible with dynamic spectrum hopping.
- **Feed-Forward DDQN (V4.0)**: Incompatible with partial observability.
- **Uncoupled Restless Bandits (Formal Whittle)**: Mathematically invalidated by cross-channel informational coupling.
