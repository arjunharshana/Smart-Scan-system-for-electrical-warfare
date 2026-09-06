# Next-Generation Benchmark Redesign Proposal: From Pattern Memorization to Cognitive Spectrum Inference

**Project**: SIH26055 / DRDO Smart Scan Strategy  
**Purpose**: Proposal for a Rigorous 4-Axis Benchmark Matrix Separating Sequence Memorization from Adaptive Temporal Inference  
**Status**: **PROPOSAL ONLY (DO NOT IMPLEMENT BEFORE FORMAL REVIEW)**  
**Date**: September 2026  

---

## 1. Motivation: The Fatal Flaws of the Current Canonical Benchmark

Our mathematical codebase audit and empirical evaluations revealed that the current 8-scenario canonical benchmark suffers from three fundamental structural weaknesses:

1. **Over-Reliance on 3-Element Circular Permutations**:  
   Scenarios 1, 2, 3, 4, and 7 all utilize subsets of the exact same 3 frequencies ($[5, 15, 25]$) out of 30 bins. 27 channels ($90\%$ of the spectrum) are perpetually vacant.
2. **Fixed Constant Dwell Times**:  
   Dwell times are integers with zero jitter ($D \in \{1, 3, 5\}$). Schedulers are never evaluated on stochastic dwell distributions or realistic pulse jitter.
3. **Episode-Stationary Regimes**:  
   No scenario alters its hopping law, dwell time, or frequency alphabet within an episode. This evaluates static zero-shot transfer, but completely fails to evaluate **real-time online cognitive adaptation under non-stationarity**.

As a consequence, algorithms that memorize fixed sequence loops appear artificially superior, while flexible adaptive algorithms are penalized if their inductive bias does not match the arbitrary cycle length.

---

## 2. The 4-Axis Benchmark Evaluation Matrix

To establish true cognitive EW scanning performance, future evaluations should draw test scenarios from a standardized 4-axis matrix:

```
                      [Axis 4: Multi-Emitter Complexity]
                                      ▲
                                      │  Dense Multi-Emitter (Interleaved)
                                      │  Multiple Independent Emitters
                                      │  Single Emitter (Baseline)
                                      │
[Axis 1: Temporal Structure] ─────────┼─────────► [Axis 3: Generalization]
  Deterministic Periodic               │             Seen Training Regime
  Stochastic Markovian                │             Unseen Permutation / Phase
  Semi-Markov (Variable Dwell)        │             Unseen Dwell Distribution
  Bursty Gated                        │             Unannounced In-Episode Shift
  Uniform Random                      │
                                      ▼
                      [Axis 2: Observation Degradation]
                         Clean (High SNR, Pd=0.98, Pfa=0.01)
                         Adverse (Fading SNR, Pd=0.80, Pfa=0.10)
                         Severe Censoring (Intermittent Blind Gaps)
```

---

### Axis 1: Temporal Process Dynamics
Evaluates the scheduler's ability to model varying degrees of temporal structure:
- **T1 — Deterministic Cyclic Loop**: Strict periodic sequence ($f_1 \to f_2 \to \dots \to f_K \to f_1$). *Baseline predictability.*
- **T2 — Stochastic 1st-Order Markov Chain**: Transitions governed by an irreducible stochastic transition matrix $\mathbf{P}$. *Evaluates conditional probability tracking.*
- **T3 — Semi-Markov with Variable Dwell**: Transitions follow a state-transition matrix $\mathbf{A}$, but dwell duration is stochastic: $d \sim \text{LogNormal}(\mu, \sigma^2)$ or $\text{Weibull}(\lambda, k)$. *Evaluates dwell duration estimation.*
- **T4 — Bursty / Gated Periodic**: Emitter transmits in bursts of duration $B$ with quiet interval $I$, hopping between pulses. *Evaluates return-period synchronization.*
- **T5 — High-Entropy Pseudo-Random**: Emitter draws hops uniformly at random from an unknown frequency dictionary. *Evaluates uniform coverage and avoidance of false pattern hallucination.*

---

### Axis 2: RF Observation Quality & Channel Degradation
Evaluates robustness to non-ideal receiver hardware and propagation environments:
- **O1 — Benign Channel**: High SNR ($>20\text{ dB}$), $P_D = 0.98$, $P_{FA} = 0.01$. Linear free-space loss.
- **O2 — Degraded Channel**: Moderate SNR ($6-10\text{ dB}$), $P_D = 0.80$, $P_{FA} = 0.08$. Rayleigh multipath fading.
- **O3 — High False Alarm Burst**: Intermittent non-Gaussian interference bursts producing localized $P_{FA} \ge 0.25$ in random bins. *Tests false alarm discrimination.*
- **O4 — Periodic Blind Gaps**: Receiver forced to tune to calibration channels for 5 consecutive steps every 50 steps. *Tests passive belief retention during sensing blackouts.*

---

### Axis 3: Operational Generalization & Adaptivity
Distinguishes pattern memorization from genuine cognitive inference:
- **G1 — In-Distribution Seen**: Exact hopping alphabet, dwell, and transition kernel seen during training. *Measures memorization ceiling.*
- **G2 — Unseen Permutation / Phase**: Known frequency alphabet, but novel hopping order and random initial phase. *Tests sequence-order invariance.*
- **G3 — Unseen Channel Alphabet**: Hopping occurs over completely novel carrier frequencies not present in the training set. *Tests channel-index decoupling.*
- **G4 — Unseen Dwell Distribution**: Same frequencies and order, but dwell shifts (e.g. trained on $D=3$, tested on $d \sim \text{DiscreteUniform}(1, 5)$). *Tests dwell flexibility.*
- **G5 — Dynamic In-Episode Regime Shift**: Emitter abruptly changes hopping alphabet or dwell duration at step $t=150$ of a 300-step episode without notification. *Tests real-time change detection and adaptation rate.*

---

### Axis 4: Spectrum Multi-Emitter Complexity
Evaluates multi-threat scalability:
- **C1 — Single Threat**: Exactly 1 emitter active in a 30-channel spectrum ($3.3\%$ occupancy).
- **C2 — Dual Asynchronous Non-Overlapping Threats**: 2 emitters hopping independently across distinct frequency bands (e.g. Threat A in bins 0–14, Threat B in bins 15–29).
- **C3 — Multi-Threat Interleaved Spectrum**: 3 emitters hopping across shared channels with pulse collisions. *Tests emitter deinterleaving under partial observability.*

---

## 3. Proposed Canonical Evaluation Suite (v5 Benchmark Matrix)

A balanced 10-scenario benchmark spanning the matrix:

| Scenario Code | Temporal Dynamic (Axis 1) | Observation Quality (Axis 2) | Generalization Dimension (Axis 3) | Multi-Emitter Complexity (Axis 4) | Evaluation Objective |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BENCH_01** | T1 (Cyclic Loop) | O1 (Benign) | G1 (In-Distribution) | C1 (Single Threat) | Historical baseline reproducibility ($[5, 15, 25]$, $D=3$). |
| **BENCH_02** | T1 (Cyclic Loop) | O1 (Benign) | G2 (Permuted Sequence) | C1 (Single Threat) | Tests sequence-order decoupling ($[25, 5, 15]$). |
| **BENCH_03** | T1 (Cyclic Loop) | O1 (Benign) | G3 (Novel Channels) | C1 (Single Threat) | Tests frequency-index independence (novel bins $[7, 18, 29]$). |
| **BENCH_04** | T3 (Semi-Markov) | O1 (Benign) | G4 (Stochastic Dwell) | C1 (Single Threat) | Dwell $d \sim \text{Uniform}\{2, 3, 4\}$. Tests dwell tracking. |
| **BENCH_05** | T2 (Markov Chain) | O1 (Benign) | G2 (Stochastic Matrix) | C1 (Single Threat) | Irreducible $4\times 4$ Markov transition matrix. |
| **BENCH_06** | T4 (Periodic Burst)| O1 (Benign) | G2 (Novel Period) | C1 (Single Threat) | Burst duration 4, Interval 18. Tests temporal PLL. |
| **BENCH_07** | T5 (Random Hopping)| O1 (Benign) | G2 (Full Entropy) | C1 (Single Threat) | Uniform random hopping. Tests pattern hallucination. |
| **BENCH_08** | T1 (Cyclic Loop) | O2 (Fading / Noise) | G2 (Permuted) | C1 (Single Threat) | $P_D = 0.80, P_{FA} = 0.08$. Tests resilience to missed hits. |
| **BENCH_09** | T1 $\to$ T3 (Shift) | O1 (Benign) | **G5 (In-Episode Shift)** | C1 (Single Threat) | **Regime shifts at step 150**. Measures recovery time. |
| **BENCH_10** | T1 (Dual Threats) | O1 (Benign) | G2 (Dual Band) | **C2 (Dual Threats)** | Threat A ($D=2$) + Threat B ($D=4$). Tests multi-threat tracking. |

---

## 4. Evaluation Protocol & New Scoring Metrics

### Adaptation Budget & Metrics
To penalize blind memorization while measuring cognitive agility:

1. **Adaptation Horizon ($T_{\text{adapt}}$)**:  
   The number of discrete steps required after episode start (or after a regime shift) for the scheduler to achieve $\text{IR} \ge 0.50$ over a rolling 30-step window.
2. **Steady-State Interception Ratio ($\text{IR}_{\text{steady}}$)**:  
   Mean interception ratio evaluated strictly over the final $60\%$ of the episode (steps $120-300$), isolating steady-state tracking from initial exploration.
3. **Cognitive Agility Index (CAI)**:  
   A composite metric balancing steady-state tracking efficiency against exploration cost:
   $$\text{CAI} = \text{IR}_{\text{steady}} \times \left(1.0 - \frac{T_{\text{adapt}}}{T_{\text{episode}}}\right) \times \left(1.0 - \frac{\text{Excess Scans on Idle Bands}}{\text{Total Steps}}\right)$$
4. **Hallucination Penalty**:  
   On random hopping (BENCH_07), any scheduler achieving $\text{IR} < 0.20$ receives an explicit penalty, penalizing algorithms that overfit to phantom patterns.

---

## 5. Implementation Roadmap (Post-Approval)

When approved by project leadership:
1. **Phase 1**: Implement stochastic dwell distribution generator in `rf_environment/frequency_behaviors/`.
2. **Phase 2**: Implement in-episode regime switching hook in `rf_environment/environment/rf_environment.py`.
3. **Phase 3**: Implement multi-threat deinterleaving observer in `rf_environment/environment/observation_builder.py`.
4. **Phase 4**: Implement the 10-scenario catalog in `benchmarks/scenarios_v5.py`.
5. **Phase 5**: Benchmark candidate algorithms (CA-Dwell, HSMM, Whittle Heuristic, and V4.1 Frozen) against the v5 matrix.
