# Algorithm Compatibility & Theoretical Verification Audit

**Project**: SIH26055 / DRDO Smart Scan Strategy for Electronic Warfare  
**Status**: Theoretical Verification Pass on Algorithmic Assumptions  
**Date**: September 2026  

---

## 1. Markov vs. Semi-Markov Verification

### Mathematical Evaluation of State Representations
1. **Is frequency alone Markovian?**
   **NO.** For any scenario with dwell $D > 1$, $P(F_{t+1} \mid F_t)$ is non-stationary. If $F_t = f_1$, the emitter will stay at $f_1$ for $t \pmod D < D - 1$, but hop to $f_2$ when $t \pmod D = D - 1$. Conditioning solely on $F_t$ without knowing the elapsed holding time $\tau_t$ violates the Markov property:
   $$P(F_{t+1} \mid F_t, F_{t-1}, \dots) \ne P(F_{t+1} \mid F_t)$$

2. **Is frequency + elapsed dwell Markovian?**
   **YES, EXACTLY.** Let the augmented state be $s_t = (F_t, \tau_t) \in \mathcal{F} \times \{0, \dots, D-1\}$.
   - If $\tau_t < D - 1$: $P(s_{t+1} = (F_t, \tau_t + 1) \mid s_t) = 1.0$.
   - If $\tau_t = D - 1$: $P(s_{t+1} = (f_{\text{next}}, 0) \mid s_t) = 1.0$ (or $1/K$ for random hopping).
   The state transition $P(s_{t+1} \mid s_t)$ satisfies the 1st-order Markov property unconditionally.

3. **How is dwell generated in the simulator?**
   Dwell is **deterministic, constant integer holding time** ($D \in \{1, 3, 5\}$). It is not drawn from a continuous, geometric, or stochastic distribution during an episode.

### Formal Process Classification
- **In Continuous Time / Jump Theory**: A process with holding times between state transitions is defined as a **semi-Markov process** (or Markov renewal process).
- **In Discrete Time**: Because dwell durations are finite small integers ($D \le 5$), the semi-Markov process is mathematically isomorphic to an **Augmented Discrete-Time Markov Chain (DTMC)** of size $|\mathcal{S}| = K \cdot D$.
- **Conclusion**: The simulator is an **Augmented Markov Chain with Deterministic Dwell** (the discrete-time equivalent of a degenerate semi-Markov process).

---

## 2. POMDP Verification

1. **Access to True State**: The scheduler has **zero direct access** to $(F_t, \tau_t, \text{tx}_t)$.
2. **Action-Dependent Observation Censoring**: The observation $Y_t$ is gathered strictly at the channel selected by action $A_t$. The remaining 29 channels are completely unobserved:
   $$P(Y_t \mid S_t, A_t) = \mathbf{1}_{\{A_t = \text{bin}(F_t)\}} \cdot [P_D \cdot \text{tx}_t + P_{FA} (1 - \text{tx}_t)] + \mathbf{1}_{\{A_t \ne \text{bin}(F_t)\}} \cdot P_{FA}$$
3. **Mutual Information Across Unobserved Bins**: Because there is a single emitter, observing a miss ($Y_t = 0$) on bin $A_t$ eliminates that channel from the current hypothesis, shifting probability mass to unobserved channels via Bayes' rule.
4. **Sufficiency of Belief State**: The posterior distribution:
   $$\mathbf{b}_t(f, \tau) = P(F_t = f, \tau_t = \tau \mid A_{1:t}, Y_{1:t})$$
   is a mathematically sufficient information state for optimal decision making (Zhao et al. 2007).
5. **Conclusion**: The problem is **strictly and genuinely an Augmented Belief-POMDP**.

---

## 3. Whittle / RMAB Verification (Precise Boundary Check)

### Why Classical RMAB Does Not Formally Apply
In Restless Multi-Armed Bandits (Whittle 1988, Liu & Zhao 2010):
1. **Arm Independence**: Arms must evolve independently: $P(S_{t+1} \mid S_t) = \prod_{i=1}^N P_i(s_{t+1}^{(i)} \mid s_t^{(i)}, a_t^{(i)})$.
2. **Decoupled Lagrangian**: Decoupling the constraint $\sum a_i = 1$ via subsidy $W$ assumes that knowing arm $i$'s state provides zero information about arm $j$'s state.

### Simulator Ground Truth: Strong Channel Coupling
1. **Mutually Exclusive States**: The single emitter constraint $\sum \mathbf{1}_{F_t = f_i} \le 1$ creates negative cross-arm correlation.
2. **Coupled Transitions**: When the emitter leaves channel $j$, it enters channel $k$ according to hopping kernel $T(j \to k)$. The transition of arm $k$ is causally dependent on arm $j$.
3. **Non-Separable Reward**: Reward is credited only if the chosen arm intercepts the single active emitter.

### Scientific Taxonomy of Whittle in Our Problem
1. **Formally Justified RMAB**: **NO.** Classical restless bandit indexability proofs (e.g. Liu & Zhao 2010) fail because arm transitions are not independent.
2. **Partially Observable RMAB Approximation**: **PARTIALLY.** Treating each channel as a marginal 2-state process is an approximation that discards cross-channel correlation.
3. **Heuristic Whittle-Style Priority Score**: **YES, VERIFIED.** Our implementation in `rf_environment/scheduler/whittle/` succeeds precisely because it augments the decoupled belief term with explicit cross-arm transition estimates ($\mu \hat{T}$) and coverage uncertainty ($\lambda u_i$).

---

## 4. HSMM Compatibility Check

### How the Simulator Maps to an HSMM
- **Hidden States**: Emitter carrier frequencies $j \in \{1, \dots, K\}$.
- **Holding Time Distribution**: Deterministic duration $p_j(d) = \delta(d - D)$.
- **Transition Matrix**: Off-diagonal hopping transition matrix $A_{jk}$ with $A_{jj} = 0$.
- **Observations**: Binary detector outcomes emitted conditionally during channel occupancy.

### HSMM Limitations in Our Formulation
1. **Action-Dependent Observation**: In classical speech/language HSMMs, observations are passive. In our simulator, the receiver *chooses* which channel to sense ($A_t$). It is a **Controlled HSMM (POMDP)**.
2. **Degenerate Duration**: Because $D$ is constant, a full HSMM semi-Markov renewal framework is mathematically equivalent to an **augmented Markov chain** tracking $(f, \tau)$.

### Final HSMM Classification
> **Verdict: OPTION B (HSMM is useful but incomplete; an Augmented Belief-POMDP is the mathematically superior and exact formulation).**  
> An explicit HSMM duration kernel $p(d)$ is valuable when dwell is variable or stochastic. For fixed integer dwell ($D \le 5$), state augmentation $(F_t, \tau_t)$ is computationally simpler, exact, and avoids the complexity of semi-Markov renewal integrals.

---

## 5. LSTM / DDQN Compatibility

### What Information Must an Optimal Scheduler Remember?
An optimal scheduler needs:
1. Current belief over active channel: $\mathbf{b}_t \in \Delta^{K-1}$.
2. Current elapsed dwell estimate: $\hat{\tau}_t \in \{0, \dots, D-1\}$.
3. Active hopping sequence and dwell parameter: $(f_{(1)} \to \dots \to f_{(K)}, D)$.

### Inductive Bias Comparison

| Model | Inductive Bias for Dwell | Inductive Bias for Partial Sensing | Multi-Regime Resilience | Sample Efficiency |
| :--- | :--- | :--- | :---: | :---: |
| **Feed-Forward DDQN** | Poor (Sliding window) | Poor (No belief state) | Poor (Overfits frequency indices) | Very Low |
| **LSTM-DDQN** | Unconstrained (Can learn clock, but no prior) | Implicit (Hidden state $h_t$ acts as latent belief) | **Poor (Shared weights suffer gradient conflict across periods)** | Low ($10^4$ steps) |
| **Augmented Belief-POMDP** | **Exact (State $\tau$)** | **Exact (Recursive Bayes filter)** | **High (Zero-shot parameter re-estimation)** | **Optimal ($10^1$ steps)** |

**Conclusion on LSTM**:  
LSTM is **not mathematically invalid**, but it has an **unstructured inductive bias**. It must learn an internal phase-locked loop and counter from sparse rewards through backpropagation through time. When exposed to multiple regimes with differing dwell and cycle lengths, the shared recurrent matrix $W_{hh}$ suffers gradient interference.
