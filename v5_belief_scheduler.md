# V5.0 Augmented Belief-State Bayesian Scheduler: Architecture, Evaluation & Scientific Report

**Project**: SIH26055 / DRDO Smart Scan Strategy for Electronic Warfare  
**Status**: Final Algorithm Implementation & Empirical Evaluation  
**Date**: September 2026  

---

## 1. Problem Formulation

The Electronic Warfare (EW) cognitive spectrum scanning problem is formulated as a discrete-time, partially observable controlled decision process. An intercept receiver with narrow instantaneous bandwidth ($M=1$ channel, $20\text{ MHz}$) must monitor a wideband RF spectrum ($N=30$ channels, $100\text{ MHz} - 700\text{ MHz}$) containing non-stationary hopping and periodic radar threats.

At each discrete time step $t \in \mathbb{N}$:
1. The scheduler selects a single channel to scan: $A_t \in \{0, 1, \dots, N-1\}$.
2. The receiver tunes to center frequency $f_{\text{rx}}(A_t)$ and obtains a noisy binary detection outcome $Y_t \in \{0, 1\}$.
3. The remaining $N - 1 = 29$ channels ($96.7\%$ of the spectrum) are completely unobserved.
4. The objective is to maximize the Interception Rate (IR) under partial observation censoring.

---

## 2. Hidden State Representation

The physical simulation state consists of:
$$S_t = (F_t, \tau_t, D_t)$$
- $F_t \in \{0, 1, \dots, 29\}$: Active emitter carrier frequency bin.
- $\tau_t \in \{0, 1, \dots, D_t - 1\}$: Elapsed dwell step within the current hop.
- $D_t \in \{1, 3, 5\}$: True ground-truth emitter dwell duration (deterministic integer constant).

The total discrete state space size is:
$$|\mathcal{S}| = \sum_{d \in \{1, 3, 5\}} 30 \times d = 30 \times (1 + 3 + 5) = 270 \text{ states}$$

---

## 3. Belief Representation

The scheduler maintains an exact recursive Bayesian belief state:
$$b_t(f, \tau, d) = P(F_t = f, \tau_t = \tau, D = d \mid A_{1:t-1}, Y_{1:t-1})$$

### Mathematical Invariants:
1. Non-negativity: $b_t(f, \tau, d) \ge 0$ for all $(f, \tau, d) \in \mathcal{S}$.
2. Total Probability Axiom: $\sum_{d \in \{1, 3, 5\}} \sum_{f=0}^{29} \sum_{\tau=0}^{d-1} b_t(f, \tau, d) \equiv 1.0$.
3. Numerical Underflow Clamping: Minimum probability floor $10^{-12}$.
4. Storage: Dict of 2D NumPy arrays `belief[d]` of shape $(30, d)$ for $d \in \{1, 3, 5\}$.

---

## 4. Transition Model

The forward semi-Markov temporal progression $b_{t+1}^- = \mathcal{T}(b_t)$ is:

### For Dwell Progression ($\tau < d - 1$):
The emitter remains on the current carrier frequency:
$$P(F_{t+1} = f, \tau_{t+1} = \tau + 1 \mid F_t = f, \tau_t = \tau) = 1.0$$

### At Hop Boundaries ($\tau = d - 1$):
The emitter completes its dwell on channel $f$ and hops to a new frequency $f_{\text{next}}$ with elapsed dwell resetting to zero:
$$P(F_{t+1} = f_{\text{next}}, \tau_{t+1} = 0 \mid F_t = f, \tau_t = d - 1) = T(f \to f_{\text{next}})$$
where $T \in \mathbb{R}^{30 \times 30}$ is the inter-frequency transition kernel.

---

## 5. Observation Likelihood

Given action $A_t$ and binary detector outcome $Y_t \in \{0, 1\}$ from the detector model ($P_D = 0.95$, $P_{FA} = 0.02$):

$$P(Y_t \mid F_t = f, A_t) = \begin{cases}
P_D^{Y_t} (1 - P_D)^{1 - Y_t} & \text{if } f = A_t \text{ (In-Band)} \\
P_{FA}^{Y_t} (1 - P_{FA})^{1 - Y_t} & \text{if } f \ne A_t \text{ (Out-of-Band)}
\end{cases}$$

---

## 6. Bayesian Update Equations

Upon observing detector outcome $Y_t$ at scanned channel $A_t$:

1. **Unnormalized Posterior**:
   $$\tilde{b}_t(f, \tau, d) = P(Y_t \mid F_t = f, A_t) \cdot b_t^-(f, \tau, d)$$

2. **Normalization**:
   $$Z_t = \sum_{d} \sum_{f} \sum_{\tau} \tilde{b}_t(f, \tau, d)$$
   $$b_t(f, \tau, d) = \frac{\tilde{b}_t(f, \tau, d)}{Z_t}$$

This update operates symmetrically on:
- **Positive Evidence ($Y=1$)**: Boosts probability on scanned bin by $\frac{P_D}{P_{FA}} = \frac{0.95}{0.02} = 47.5\times$.
- **Negative Evidence ($Y=0$)**: Suppresses probability on scanned bin by $\frac{1 - P_D}{1 - P_{FA}} = \frac{0.05}{0.98} \approx 0.051\times$.

---

## 7. Dwell Inference

The marginal distribution over candidate dwell values is obtained via Bayesian Model Averaging (BMA):
$$P(D = d) = \sum_{f=0}^{29} \sum_{\tau=0}^{d-1} b_t(f, \tau, d)$$

- If an emitter remains detected on a channel for multiple consecutive steps, likelihood evidence flows preferentially into hypotheses with $D > 1$, naturally suppressing the $D=1$ model.
- If an emitter hops every step, hypotheses with $D > 1$ suffer repeated observation misses, shifting mass into $D=1$.

---

## 8. Hop-Transition Inference

In **Ablation B2**, the transition kernel $T(i \to j)$ is learned online from observed detector confirmations without accessing ground truth:
$$\hat{T}(i \to j) = \frac{C[i, j] + \alpha}{\sum_k (C[i, k] + \alpha)}$$
where $C[i, j]$ counts observed successive detections transitioning from bin $i$ to bin $j$, and $\alpha = 0.05$ is a Dirichlet smoothing prior.

---

## 9. Action-Selection Rule

The scheduler computes marginal channel occupancy probabilities:
$$P(F_t = f) = \sum_{d \in \{1, 3, 5\}} \sum_{\tau=0}^{d-1} b_t(f, \tau, d)$$

Under `exploration_mode = "uncertainty"`, an information coverage bonus $u_f(t)$ is added:
$$u_f(t) = \min\left(1.0, \frac{\tau_f^{\text{scan}}}{K_{\text{stale}}}\right)$$
$$\text{Score}(f) = P(F_t = f) + \lambda \cdot u_f(t)$$
$$A_t = \arg\max_f \text{Score}(f)$$
Ties are resolved deterministically to the lowest channel index.

---

## 10. Ground-Truth Isolation

- The V5.0 scheduler consumes strictly `SchedulerObservation`.
- It has **zero access** to `ground_truth`, scenario dictionaries, true emitter frequencies, or scenario dwell values.
- Automated code inspection (`test_sanity_6_gt_isolation`) confirms that no forbidden ground-truth references exist in the implementation.

---

## 11. Computational Complexity & Latency

Evaluated over 500 active scanning steps:
- **State Space**: 270 discrete states.
- **Decision Latency**: **$35.0\ \mu\text{s}$ per step** (Mean across benchmark: $350\ \mu\text{s}$ including full Python environment step).
- **Throughput**: $> 28,000$ decisions/second on single CPU core.
- **Hardware**: CPU only (Zero GPU requirements).
- **Training**: 0.0 seconds (Zero offline training).

---

## 12. Benchmark Results & Controlled Ablations

Evaluated across all 8 Canonical Test Scenarios over 5 random seeds (300 steps per episode):

| Canonical Test Scenario | Threat Profile | B0: Frequency Only | B1: Augmented Belief (Uniform Hop) | B2: Augmented Belief (Learned Hop) |
| :--- | :--- | :---: | :---: | :---: |
| **1_Seen_Structure** | $[5, 15, 25]$, dwell=3 | $11.20\% \pm 0.75\%$ | $8.60\% \pm 4.22\%$ | **$21.20\% \pm 8.33\%$** |
| **2_Unseen_Permutation** | $[25, 5, 15]$, dwell=3 | $12.20\% \pm 0.75\%$ | $11.00\% \pm 1.67\%$ | **$18.20\% \pm 3.19\%$** |
| **3_Unseen_Phase** | $[15, 25, 5]$, dwell=3 | $11.20\% \pm 0.75\%$ | $11.60\% \pm 3.01\%$ | **$19.40\% \pm 7.55\%$** |
| **4_Unseen_Dwell** | $[5, 15, 25]$, dwell=1 | $3.73\% \pm 0.25\%$ | $2.53\% \pm 0.50\%$ | **$11.47\% \pm 6.75\%$** |
| **5_Unseen_Subset** | $[10, 20, 27]$, dwell=3 | $6.40\% \pm 0.49\%$ | $5.00\% \pm 2.10\%$ | **$10.40\% \pm 4.50\%$** |
| **6_Mixed_Shift** | $[10, 27, 20]$, dwell=5 | $7.67\% \pm 0.82\%$ | $3.00\% \pm 1.63\%$ | **$8.00\% \pm 2.45\%$** |
| **7_Random_Hopping** | $[5, 15, 25]$, random | $3.87\% \pm 0.69\%$ | $3.07\% \pm 1.50\%$ | **$9.20\% \pm 4.75\%$** |
| **8_Periodic_Burst** | Bin 15, 4 on / 11 off | $29.00\% \pm 2.00\%$ | $10.00\% \pm 3.16\%$ | **$25.00\% \pm 16.73\%$** |
| **OVERALL MEAN IR** | — | **10.66%** | **6.85%** | **15.36%** |

### Key Ablation Insights:
1. **B1 vs B0 ($6.85\%$ vs $10.66\%$)**: Assuming uniform transitions when hopping causes probability mass to diffuse equally into 29 other channels ($1/29 \approx 0.034$ each), losing track of the emitter.
2. **B2 vs B1 ($15.36\%$ vs $6.85\%$)**: Online empirical transition learning more than **doubles interception rate** by concentrating transition mass into confirmed historical transitions ($5 \to 15 \to 25 \to 5$).

---

## 13. Comparison with Previous Schedulers

| Scenario | Threat Profile | Context-Aware (CA) | V4.0 Hybrid | V4.1 LSTM ($H=64$) | Whittle W3 | V5.0 Belief (B2) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1_Seen_Structure** | $[5, 15, 25]$, dwell=3 | $26.90\%$ | $\mathbf{50.62\%}$ | $34.80\%$ | $29.90\%$ | $21.20\%$ |
| **2_Unseen_Permutation** | $[25, 5, 15]$, dwell=3 | $31.40\%$ | $40.48\%$ | $28.70\%$ | $\mathbf{46.80\%}$ | $18.20\%$ |
| **3_Unseen_Phase** | $[15, 25, 5]$, dwell=3 | $30.40\%$ | $\mathbf{45.70\%}$ | $32.60\%$ | $38.50\%$ | $19.40\%$ |
| **4_Unseen_Dwell** | $[5, 15, 25]$, dwell=1 | $\mathbf{24.53\%}$ | $19.57\%$ | $18.30\%$ | $8.43\%$ | $11.47\%$ |
| **5_Unseen_Subset** | $[10, 20, 27]$, dwell=3 | $27.40\%$ | $21.42\%$ | $22.10\%$ | $\mathbf{29.20\%}$ | $10.40\%$ |
| **6_Mixed_Shift** | $[10, 27, 20]$, dwell=5 | $\mathbf{30.67\%}$ | $14.71\%$ | $22.00\%$ | $9.17\%$ | $8.00\%$ |
| **7_Random_Hopping** | $[5, 15, 25]$, random | $\mathbf{26.37\%}$ | $14.18\%$ | $15.20\%$ | $26.17\%$ | $9.20\%$ |
| **8_Periodic_Burst** | Bin 15, 4 on / 11 off | $51.50\%$ | $74.83\%$ | $41.50\%$ | $\mathbf{87.00\%}$ | $25.00\%$ |
| **OVERALL MEAN IR** | — | **31.15%** | $\mathbf{35.19\%}$ | **26.90%** | **34.40%** | **15.36%** |

### Computational & Architectural Summary:

| Scheduler | Overall IR | Runtime/step | Training Required | GT Access | Model Nature |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Context-Aware (CA)** | $31.15\%$ | $53.3\ \mu\text{s}$ | **No** | **No** | Statistical Markov + Activity + Staleness |
| **V4.0 Hybrid** | $\mathbf{35.19\%}$ | $85.0\ \mu\text{s}$ | Yes (Offline) | **No** | CA + Feedforward DDQN + Meta-Arbitrator |
| **V4.1 LSTM-Hybrid** | $26.90\%$ | $110.8\ \mu\text{s}$ | Yes (Offline) | **No** | CA + LSTM-DDQN ($H=64$) + Meta-Arbitrator |
| **Whittle-Style (W3)** | **$34.40\%$** | $162.9\ \mu\text{s}$ | **No** | **No** | Restless Bandit Index + Dwell + Periodicity |
| **V5.0 Bayesian Belief** | $15.36\%$ | $35.0\ \mu\text{s}$ | **No** | **No** | Pure Augmented POMDP Belief State ($F, \tau, D$) |

---

## 14. Scientific Diagnosis & Limitations of Pure POMDP

### Why Did V5.0 Not Beat Whittle W3 or Context-Aware?

The empirical results answer the core scientific question of this investigation:
> **Does a pure, unconstrained Bayesian Belief-POMDP over $(F_t, \tau_t, D)$ outperform decoupled heuristic index schedulers in wideband EW search?**  
> **Answer: NO.**

### Fundamental Root Causes:
1. **The Curse of Wideband Spatial Sparsity**:
   In 30 channels with only 3 active frequencies ($10\%$ occupancy), a receiver sensing only 1 channel leaves 29 channels ($96.7\%$) completely dark. In a pure POMDP, a miss ($Y=0$) on channel $j$ reduces $b(j)$ but spreads probability uniformly across all other 29 channels ($+0.034$ per channel). When an emitter hops, the belief state across unobserved channels is flat, forcing the scheduler into slow round-robin search.
2. **Why CA and Whittle Succeeded**:
   - **Context-Aware (CA)** maintains an **Activity Vector $A_j$** that tracks which channels historically produced detections. When an emitter hops, CA does not search all 29 channels; it prioritizes channels with high historical activity ($5, 15, 25$), effectively reducing the search space from 30 channels to 3 channels!
   - **Whittle W3** maintains an explicit **Periodicity Term $V_{\text{periodic}}$** that synchronizes with the return interval $\Delta t \approx 9$.
3. **Pure Belief vs. Activity-Biased Indexing**:
   A mathematically pure POMDP belief distribution does not inherently maintain an active channel alphabet unless formulated over the combinatorial power set of channel subsets ($\binom{30}{3} = 4,060$ hypotheses).

---

## 15. Final Conclusion & Recommendation

In accordance with the project directives:
- **Algorithm development is now officially complete.**
- The empirical evidence conclusively establishes that **Whittle-Style Heuristic Index Scheduler (W3)** (34.40% IR, 87% on burst radar, zero training, zero GT) and **Context-Aware (CA)** (31.15% IR) are the most robust, computationally efficient, and tactically resilient scan schedulers for this problem.
- V5.0 serves as a rigorous scientific benchmark demonstrating the limits of passive Bayesian filtering under extreme spatial observation censoring.
