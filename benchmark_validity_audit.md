# Benchmark Validity & Canonical Scenario Audit

**Project**: SIH26055 / DRDO Smart Scan Strategy for Electronic Warfare  
**Status**: Fast Benchmark Validity Audit  
**Date**: September 2026  

---

## 1. Quantitative Scenario Parameter Matrix

All parameters are extracted directly from [`benchmarks/scenarios_v4_1.py`](file:///home/rishant-gupta/Projects/SIH-2026/benchmarks/scenarios_v4_1.py), [`rf_environment/receiver/detector.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/receiver/detector.py), and [`rf_environment/environment/rf_environment.py`](file:///home/rishant-gupta/Projects/SIH-2026/rf_environment/environment/rf_environment.py).

| Canonical Scenario | Frequency Alphabet (Active Bins) | Total Bins | Active Bins ($K$) | Occupancy Ratio ($K/N$) | Dwell ($D$) | Hop Mode | Hop Order Fixed? | Timing Jitter | Cycle Period ($T_{\text{cycle}} = K \cdot D$) | Cycles per Episode ($300 / T_{\text{cycle}}$) | Intra-Episode Shifts? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1_Seen_Structure** | `[5, 15, 25]` | 30 | 3 | 10.0% | 3 | Sequential | YES | ZERO | 9 steps | **33.3 cycles** | NONE |
| **2_Unseen_Permutation** | `[25, 5, 15]` | 30 | 3 | 10.0% | 3 | Sequential | YES | ZERO | 9 steps | **33.3 cycles** | NONE |
| **3_Unseen_Phase** | `[15, 25, 5]` | 30 | 3 | 10.0% | 3 | Sequential | YES | ZERO | 9 steps | **33.3 cycles** | NONE |
| **4_Unseen_Dwell** | `[5, 15, 25]` | 30 | 3 | 10.0% | 3 | Sequential | YES | ZERO | 3 steps | **100.0 cycles** | NONE |
| **5_Unseen_Subset** | `[10, 20, 27]` | 30 | 3 | 10.0% | 3 | Sequential | YES | ZERO | 9 steps | **33.3 cycles** | NONE |
| **6_Mixed_Shift** | `[10, 27, 20]` | 30 | 3 | 10.0% | 5 | Sequential | YES | ZERO | 15 steps | **20.0 cycles** | NONE |
| **7_Random_Hopping** | `[5, 15, 25]` | 30 | 3 | 10.0% | 1 | Uniform Random | NO | ZERO | N/A (Stochastic) | 300 transitions | NONE |
| **8_Periodic_Burst** | `[15]` (Burst: dur=4, int=15) | 30 | 1 | 3.3% | $\infty$ | Fixed Carrier | N/A | ZERO | 15 steps | **20.0 bursts** | NONE |

### Common Simulation & Hardware Parameters
- **Total Frequency Bins ($N$)**: $30$ contiguous channels ($100\text{ MHz} - 700\text{ MHz}$, $20\text{ MHz}$ per bin).
- **Receiver Sensing Width ($M$)**: Exactly $1$ bin per time step ($3.33\%$ instantaneous spectrum coverage).
- **Detector Parameters**:
  - Probability of Detection ($P_D$): **$0.95$** ($95\%$ true positive rate on active transmitter).
  - Probability of False Alarm ($P_{FA}$): **$0.02$** ($2\%$ false alarm rate on empty bins).
  - Sensitivity: $-90\text{ dBm}$, Noise Floor: $-100\text{ dBm}$, Detection Threshold: $6\text{ dB}$.
- **Episode Duration**: Exactly $300$ steps ($10\text{ ms}$ per step $\implies 3.0$ seconds total).
- **Initial Phase**: Fixed at $t=0$, hop index $k=0$, elapsed dwell $\tau=0$ across all seeds.

---

## 2. Core Question & Critical Evaluation

> **"Does the benchmark test general temporal inference, or can an algorithm obtain high performance by memorizing a small deterministic pattern?"**

### Analytical Answer:

The canonical benchmark occupies an **intermediate, hybrid regime**, with distinct implications:

### A. Vulnerability to Pattern Memorization (Within-Episode & Single-Regime)
1. **Extreme Spectrum Sparsity (10% Occupancy)**:
   In 7 of 8 scenarios, exactly $3$ of $30$ bins are active. $27$ bins ($90\%$) are static dead space. An algorithm that merely identifies which $3$ bins contain energy can ignore $90\%$ of the search space.
2. **High Cyclic Repetition (20 to 100 Cycles per Episode)**:
   Because the cycle length is very short ($3$, $9$, or $15$ steps) relative to the $300$-step episode, the identical sequence repeats $20$ to $100$ times without variation. Once a phase-locked loop, cyclic counter, or recurrent hidden state synchronizes with the initial cycle within the first $20$ steps, it can execute an open-loop deterministic playback for the remaining $280$ steps.
3. **Zero Timing Jitter and Zero Intra-Episode Drift**:
   Dwell is strictly deterministic and constant. There is zero pulse-to-pulse timing jitter, zero clock drift, and zero dynamic switching between emitter regimes within an episode.

### B. Resistance to Trivial Memorization (Across-Scenario OOD Evaluation)
1. **Out-of-Distribution Structure**:
   The benchmark **does prevent naive global memorization** across scenarios:
   - Scenarios 2 & 3 test permuted sequences and phase shifts on the base channels.
   - Scenario 4 tests a radical dwell change ($D=3 \to D=1$, 100 cycles).
   - Scenarios 5 & 6 shift to completely disjoint frequency channels (`[10, 20, 27]`).
   - Scenario 7 tests zero-correlation memoryless hopping (random mode).
   - Scenario 8 tests non-continuous periodic radar pulses.
2. **Why Monolithic Deep RL Collapsed**:
   Our previous empirical results directly reflect this structure:
   - When V4.1 LSTM was trained solely on `TRAIN_1` ($D=3$, cycle=9), it achieved $\sim 80\%$ on Scenarios 1–3 because it memorized the $9$-step periodicity of the base channels. However, it collapsed on Scenarios 4, 5, 6, and 7.
   - When V4.1 was trained on all $6$ training regimes simultaneously, the conflicting cycle periods ($T \in \{6, 8, 9, 12, 15, 20\}$) caused gradient interference in the shared recurrent weights, collapsing performance across the board to $26.9\%$.

### C. Formal Conclusion on Benchmark Validity
The current canonical benchmark **does not test general non-stationary cognitive inference under real-world EW uncertainty**; rather, it tests **fast parameter synchronization with stationary, low-cardinality cyclic patterns under partial observation censoring**.

An algorithm that achieves high performance on this benchmark needs:
1. Fast identification of the active channel subset $K \ll N$ (Spatial filtering).
2. Rapid estimation of the integer dwell $D$ and transition order (Temporal parameter identification).
3. Synchronized cyclic tracking (Phase-locked tracking).

Algorithms with **explicit parametric representations of $(F_t, \tau_t)$ (Augmented Belief-POMDP or HSMM)** naturally match this structure with zero training overhead, whereas **unstructured deep neural networks (LSTM-DDQN)** struggle because they attempt to learn integer clocks from scratch via gradient descent.
