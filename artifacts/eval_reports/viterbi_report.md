# JurisGraph Sequence-Smoothing Experiment Report

**Algorithm:** Document-Level Log-Space Viterbi Smoothing with Class Prior Balancing  
**Objective:** $\max_{y} \sum_{i=1}^M [ \log p_i(y_i) + \beta \log T(y_{i-1}, y_i) - \gamma \log \text{prior}(y_i) ]$  
**Optimal Parameters (chosen on Validation):** $\beta = 0.5$, $\gamma = 0.25$  

---

## 1. Executive Summary

| Metric | Plain Argmax | Viterbi Smoothed | Absolute Delta | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Val Macro F1** | 0.5572 | **0.6385** | **+0.0813** | +14.6% |
| **Test Macro F1** | 0.5016 | **0.5752** | **+0.0736** | +14.7% |
| **Test Accuracy** | 52.25% | **57.47%** | **+5.22%** | - |
| **Mean Chunks / Doc** | 105.8 | **27.3** (Ground Truth: 40.6) | **-78.5** | -74.2% fragmentation |

> **Threshold Verification:** The smoothing gain on the held-out LegalSeg test set is **+0.0736 (+7.36% absolute Macro F1)**, which is well above the +0.02 threshold.

---

## 2. Validation Grid Search Results (32 Combinations)

Selected parameter pair: **$\beta = 0.5, \gamma = 0.25$** (highlighted in bold).

| $\gamma$ (Prior Penalty) | $\beta$ (Transition Weight) | Validation Macro F1 | Validation Accuracy |
| :---: | :---: | :---: | :---: |
| 0 | 0 | 0.557184 | 60.33% |
| 0 | 0.25 | 0.614737 | 64.73% |
| 0 | 0.5 | 0.616933 | 64.25% |
| 0 | 0.75 | 0.587585 | 62.52% |
| 0 | 1 | 0.560130 | 61.42% |
| 0 | 1.5 | 0.504219 | 59.65% |
| 0 | 2 | 0.453101 | 57.90% |
| 0 | 3 | 0.377143 | 55.12% |
| 0.25 | 0 | 0.539616 | 57.98% |
| 0.25 | 0.25 | 0.618583 | 64.80% |
| **0.25** | **0.5** | **0.638488** (Best) | **65.41%** |
| 0.25 | 0.75 | 0.628684 | 64.25% |
| 0.25 | 1 | 0.613864 | 63.62% |
| 0.25 | 1.5 | 0.575786 | 62.53% |
| 0.25 | 2 | 0.530158 | 61.13% |
| 0.25 | 3 | 0.457184 | 58.52% |
| 0.5 | 0 | 0.497989 | 52.32% |
| 0.5 | 0.25 | 0.596748 | 62.58% |
| 0.5 | 0.5 | 0.623233 | 63.71% |
| 0.5 | 0.75 | 0.632580 | 63.83% |
| 0.5 | 1 | 0.633448 | 64.00% |
| 0.5 | 1.5 | 0.618608 | 63.60% |
| 0.5 | 2 | 0.583087 | 62.95% |
| 0.5 | 3 | 0.521515 | 61.02% |
| 1 | 0 | 0.391885 | 40.72% |
| 1 | 0.25 | 0.466900 | 48.27% |
| 1 | 0.5 | 0.497615 | 51.16% |
| 1 | 0.75 | 0.514673 | 52.23% |
| 1 | 1 | 0.528111 | 53.09% |
| 1 | 1.5 | 0.540923 | 54.18% |
| 1 | 2 | 0.538720 | 54.49% |
| 1 | 3 | 0.525035 | 55.24% |

---

## 3. Training-Label Statistics (LegalSeg `train.csv`)

- **Contiguous Same-Label Probability:** **86.20%** of sentences retain the exact rhetorical role of the preceding sentence.
- **Mean Run Lengths per Role:**
  - Facts: 8.52 sentences
  - Issue: 1.65 sentences
  - Arguments of Petitioner: 4.81 sentences
  - Arguments of Respondent: 4.57 sentences
  - Reasoning: 6.20 sentences
  - Decision: 2.87 sentences
  - None: 8.93 sentences

### Transition Matrix $T$ (Row-Normalized, Laplace Smoothed $\alpha=1.0$)

| True Role | [0] Facts | [1] Issue | [2] Argume | [3] Argume | [4] Reason | [5] Decisi | [6] None |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [0] Facts | 88.3% |  0.5% |  1.4% |  0.2% |  0.4% |  0.0% |  9.2% |
| [1] Issue | 19.7% | 39.5% |  3.4% |  0.8% |  5.4% |  0.3% | 30.9% |
| [2] Arguments of Petitioner |  0.5% |  0.2% | 79.2% |  5.5% |  2.4% |  0.1% | 12.1% |
| [3] Arguments of Respondent |  0.3% |  0.6% |  1.6% | 78.1% |  3.5% |  0.2% | 15.6% |
| [4] Reasoning |  0.2% |  0.3% |  0.7% |  0.5% | 83.9% |  1.8% | 12.5% |
| [5] Decision |  0.5% |  0.6% |  0.3% |  0.2% |  1.2% | 78.8% | 18.3% |
| [6] None |  2.7% |  0.8% |  1.4% |  1.0% |  4.6% |  0.5% | 89.0% |

### Document-Start Distribution $\pi$

| Rhetorical Role | Initial Sentence Probability $\pi(y_1)$ | Class Prior $\text{prior}(y)$ |
| :--- | :---: | :---: |
| Facts |  5.27% | 15.11% |
| Issue | 15.91% |  1.14% |
| Arguments of Petitioner |  0.10% |  5.79% |
| Arguments of Respondent |  0.22% |  4.46% |
| Reasoning |  0.38% | 18.03% |
| Decision |  0.16% |  1.74% |
| None | 77.96% | 53.72% |

---

## 4. Top-k Recall on Held-Out Splits

Fraction of true-class sentences where the true role is within the top-$k$ model probabilities:

| Rhetorical Role | Val Top-1 | Val Top-2 | Val Top-3 | Test Top-1 | Test Top-2 | Test Top-3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Facts | 69.83% | 82.02% | 90.38% | 72.31% | 83.62% | 91.61% |
| Issue | 63.07% | 76.74% | 84.33% | 51.80% | 66.01% | 78.16% |
| Arguments of Petitioner | 39.64% | 70.98% | 81.01% | 37.42% | 63.40% | 74.59% |
| Arguments of Respondent | 42.53% | 63.25% | 76.80% | 34.82% | 55.95% | 69.65% |
| Reasoning | 47.03% | 81.70% | 93.38% | 43.24% | 76.48% | 91.24% |
| Decision | 75.52% | 80.00% | 84.15% | 63.58% | 70.09% | 77.04% |
| None | 69.10% | 92.50% | 98.08% | 55.16% | 88.62% | 97.13% |

---

## 5. Per-Class Test Performance Comparison

| Rhetorical Role | Argmax Precision | Argmax Recall | Argmax F1 | Viterbi Precision | Viterbi Recall | Viterbi F1 | $\Delta$ F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Facts** | 0.4804 | 0.7231 | 0.5773 | 0.5387 | 0.8123 | **0.6478** | **+0.0705** |
| **Issue** | 0.4570 | 0.5180 | 0.4856 | 0.5801 | 0.4733 | **0.5213** | **+0.0357** |
| **Arguments of Petitioner** | 0.4539 | 0.3745 | 0.4104 | 0.5895 | 0.5314 | **0.5589** | **+0.1485** |
| **Arguments of Respondent** | 0.3783 | 0.3481 | 0.3625 | 0.5595 | 0.4388 | **0.4918** | **+0.1293** |
| **Reasoning** | 0.4951 | 0.4332 | 0.4621 | 0.5541 | 0.4785 | **0.5136** | **+0.0515** |
| **Decision** | 0.6404 | 0.6358 | 0.6381 | 0.7212 | 0.7050 | **0.7130** | **+0.0749** |
| **None** | 0.6017 | 0.5512 | 0.5753 | 0.5997 | 0.5615 | **0.5799** | **+0.0046** |
| **Macro Average** | 0.5010 | 0.5120 | 0.5016 | 0.5918 | 0.5715 | **0.5752** | **+0.0736** |
| **Weighted Average** | 0.5260 | 0.5225 | 0.5189 | 0.5777 | 0.5747 | **0.5700** | **+0.0511** |

---

## 6. Row-Normalized Confusion Matrix (Viterbi on Test)

Rows: True Class | Columns: Predicted Class (Diagonal = Recall)

| True Role | [0] Facts | [1] Issue | [2] Argume | [3] Argume | [4] Reason | [5] Decisi | [6] None |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [0] Facts | 81.2% |  0.6% |  2.0% |  0.8% |  2.3% |  0.6% | 12.4% |
| [1] Issue | 37.4% | 47.3% |  1.2% |  0.1% |  2.0% |  0.9% | 11.1% |
| [2] Arguments of Petitioner |  9.2% |  0.2% | 53.1% | 10.5% |  6.2% |  0.1% | 20.6% |
| [3] Arguments of Respondent | 11.5% |  0.1% | 20.9% | 43.9% |  8.3% |  0.2% | 15.2% |
| [4] Reasoning | 10.1% |  0.2% |  2.2% |  1.4% | 47.9% |  1.1% | 37.2% |
| [5] Decision |  5.5% |  0.5% |  0.9% |  0.5% |  8.0% | 70.5% | 14.1% |
| [6] None | 17.6% |  0.6% |  3.4% |  1.8% | 19.7% |  0.8% | 56.1% |

---

## 7. Contiguous Document Chunking Impact

- **Ground Truth Chunks / Document:** **40.56**
- **Plain Argmax Chunks / Document:** **105.83** (severe over-fragmentation due to per-sentence classification noise)
- **Viterbi Smoothed Chunks / Document:** **27.31** (eliminates spurious 1-sentence micro-switches and coalesces sustained rhetorical sections)
