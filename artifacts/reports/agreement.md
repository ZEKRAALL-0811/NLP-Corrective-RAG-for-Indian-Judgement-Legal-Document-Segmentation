# Primary vs Secondary Judge Agreement Report

- **Primary Judge**: `gemini-3.5-flash-lite`
- **Secondary Judge**: `gemma-4-26b-a4b-it`
- **Last Updated**: `2026-10-20 23:54:04`

## 1. Judge Audit Agreement (6 Cases x 4 Faithfulness Variants)

| Case ID | Variant | Primary Strict Score | Secondary Strict Score | Absolute Difference | Binary Match |
|---|---|---|---|---|---|
| `gold_1` | `a_faithful` | 1.00 | 1.00 | 0.00 | MATCH |
| `gold_1` | `b_number_changed` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_1` | `c_diff_chunk` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_1` | `d_conclusion_reversed` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_3` | `a_faithful` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_3` | `b_number_changed` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_3` | `c_diff_chunk` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_3` | `d_conclusion_reversed` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_5` | `a_faithful` | 0.6666666666666666 | N/A | N/A | SKIPPED |
| `gold_5` | `b_number_changed` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_5` | `c_diff_chunk` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_5` | `d_conclusion_reversed` | 0.0 | N/A | N/A | SKIPPED |
| `gold_12` | `a_faithful` | 0.8571428571428571 | N/A | N/A | SKIPPED |
| `gold_12` | `b_number_changed` | 0.0 | N/A | N/A | SKIPPED |
| `gold_12` | `c_diff_chunk` | 0.0 | N/A | N/A | SKIPPED |
| `gold_12` | `d_conclusion_reversed` | 0.0 | N/A | N/A | SKIPPED |
| `gold_14` | `a_faithful` | 1.00 | 1.00 | 0.00 | MATCH |
| `gold_14` | `b_number_changed` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_14` | `c_diff_chunk` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_14` | `d_conclusion_reversed` | 0.00 | 0.00 | 0.00 | MATCH |
| `gold_16` | `a_faithful` | 0.5 | N/A | N/A | SKIPPED |
| `gold_16` | `b_number_changed` | 0.0 | N/A | N/A | SKIPPED |
| `gold_16` | `c_diff_chunk` | 0.0 | N/A | N/A | SKIPPED |
| `gold_16` | `d_conclusion_reversed` | 0.0 | N/A | N/A | SKIPPED |

**Audit Summary**: Valid Comparisons = `14/24`, Mean Absolute Difference = `0.000`, Binary Agreement = `14/14 (100.0%)`

## 2. Benchmark Random Subset Agreement (10 Shared Cases)

Evaluated on the 10 shared benchmark cases selected from Config C non-refusals.

| Case ID | Faithfulness (Primary) | Faithfulness (Secondary) | Relevancy (Primary) | Relevancy (Secondary) | Faith Diff | Rel Diff |
|---|---|---|---|---|---|---|
| `gold_1` | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| `gold_3` | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| `gold_4` | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| `gold_6` | 0.50 | 1.00 | 0.40 | 0.50 | 0.50 | 0.10 |
| `gold_7` | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| `gold_10` | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `gold_12` | 0.80 | 1.00 | 1.00 | 1.00 | 0.20 | 0.00 |
| `gold_15` | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| `gold_17` | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| `gold_21` | 1.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.00 |

**Benchmark Summary**: Valid Cases = `10/10`, Faithfulness MAD = `0.070`, Answer Relevancy MAD = `0.010`
