# DeepEval Judge Sensitivity and Grounding Audit (Standard vs Strict)

- **Primary Judge**: `gemini-3.5-flash-lite`
- **Secondary Judge**: `gemma-4-26b-a4b-it`
- **Scope**: 6 Real Answerable Cases x 5 Systematic Variants = 30 Audit Tests

## 1. Audit Variants Invariants
1. **(a) Faithful Answer**: Accurate answer grounded in Chroma context. Expected: Faithfulness >= 0.7.
2. **(b) Number/Date Changed**: Hallucinated numeric values. Expected: Faithfulness drop < 0.7.
3. **(c) Different Chunk Answer**: Unrelated factual text from another section. Expected: Faithfulness drop < 0.7.
4. **(d) Conclusion Reversed**: Legal outcome inverted (allowed <-> dismissed). Expected: Faithfulness drop < 0.7.
5. **(e) Off-Topic**: Irrelevant response evaluated with AnswerRelevancy. Expected: Relevancy drop < 0.5.

## 2. Detection Rate Comparison: Standard vs Strict Scoring

| Corruption Variant | Standard DeepEval Detection Rate | Strict Metric Primary (`flash-lite`) | Strict Metric Secondary (`gemma-26b`) |
|---|---|---|---|
| **(b) Number / Date Changed** | 3/6 (50.0%) | 6/6 (100.0%) | 4/6 (66.7%) |
| **(c) Different Chunk Answer** | 0/6 (0.0%) | **6/6 (100.0%)** | **4/6 (66.7%)** |
| **(d) Conclusion Reversed** | 4/6 (66.7%) | 6/6 (100.0%) | 3/6 (50.0%) |
| **(e) Off-Topic (Relevancy)** | 6/6 (100.0%) | 6/6 (100.0%) | 6/6 (100.0%) |

## 3. Comprehensive Audit Table (30 Variants)

| Case ID | Variant | Standard Score (Primary) | Strict Score (Primary) | Strict Score (Secondary) | Verdicts (Primary) |
|---|---|---|---|---|---|
| `gold_1` | `a_faithful` | 1.00 | **1.00** | 1.00 | `['yes', 'yes', 'yes']` |
| `gold_1` | `b_number_changed` | 0.00 | **0.00** | 0.00 | `['no', 'no', 'no']` |
| `gold_1` | `c_diff_chunk` | 1.00 | **0.00** | 0.00 | `['idk']` |
| `gold_1` | `d_conclusion_reversed` | 0.50 | **0.00** | 0.00 | `['no', 'idk']` |
| `gold_1` | `e_off_topic` | 0.00 | 0.00 | 0.00 | ['off_topic'] |
| `gold_3` | `a_faithful` | 1.00 | **0.00** | 0.00 | `['no']` |
| `gold_3` | `b_number_changed` | 0.00 | **0.00** | 0.00 | `['no']` |
| `gold_3` | `c_diff_chunk` | 1.00 | **0.00** | 0.00 | `['idk', 'idk']` |
| `gold_3` | `d_conclusion_reversed` | 1.00 | **0.00** | 0.00 | `['idk']` |
| `gold_3` | `e_off_topic` | 0.00 | 0.00 | 0.00 | ['off_topic'] |
| `gold_5` | `a_faithful` | 0.67 | **0.67** | N/A | `['yes', 'no', 'yes']` |
| `gold_5` | `b_number_changed` | 1.00 | **0.00** | 0.00 | `['no']` |
| `gold_5` | `c_diff_chunk` | 1.00 | **0.00** | 0.00 | `['idk']` |
| `gold_5` | `d_conclusion_reversed` | 1.00 | **0.00** | N/A | `['idk']` |
| `gold_5` | `e_off_topic` | 0.00 | 0.00 | 0.00 | ['off_topic'] |
| `gold_12` | `a_faithful` | 0.80 | **0.86** | N/A | `['yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes']` |
| `gold_12` | `b_number_changed` | 0.00 | **0.00** | N/A | `['no']` |
| `gold_12` | `c_diff_chunk` | 1.00 | **0.00** | N/A | `['idk']` |
| `gold_12` | `d_conclusion_reversed` | 0.00 | **0.00** | N/A | `['no', 'no']` |
| `gold_12` | `e_off_topic` | 0.00 | 0.00 | 0.00 | ['off_topic'] |
| `gold_14` | `a_faithful` | 1.00 | **1.00** | 1.00 | `['yes']` |
| `gold_14` | `b_number_changed` | 1.00 | **0.00** | 0.00 | `['no']` |
| `gold_14` | `c_diff_chunk` | 1.00 | **0.00** | 0.00 | `['idk', 'idk']` |
| `gold_14` | `d_conclusion_reversed` | 0.50 | **0.00** | 0.00 | `['no', 'idk']` |
| `gold_14` | `e_off_topic` | 0.00 | 0.00 | 0.00 | ['off_topic'] |
| `gold_16` | `a_faithful` | 0.50 | **0.50** | N/A | `['no', 'yes']` |
| `gold_16` | `b_number_changed` | 1.00 | **0.00** | N/A | `['idk', 'idk']` |
| `gold_16` | `c_diff_chunk` | 1.00 | **0.00** | N/A | `['idk']` |
| `gold_16` | `d_conclusion_reversed` | 0.00 | **0.00** | N/A | `['no']` |
| `gold_16` | `e_off_topic` | 0.00 | 0.00 | 0.00 | ['off_topic'] |
