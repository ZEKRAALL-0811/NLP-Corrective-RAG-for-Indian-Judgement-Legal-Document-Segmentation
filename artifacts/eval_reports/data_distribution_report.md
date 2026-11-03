# LegalSeg Dataset Validation & Class Distribution Report

### 1. Split Row Counts & Data Cleaning Summary

| Split | Raw Rows | Dropped (Null/Blank) | Valid Clean Rows | Retained % |
|---|---|---|---|---|
| **Train** | 1,122,507 | 131 | 1,122,376 | 99.99% |
| **Val** | 293,370 | 8 | 293,362 | 100.0% |
| **Test** | 149,881 | 13 | 149,868 | 99.99% |
| **Total** | **1,565,758** | **152** | **1,565,606** | **99.99%** |

### 2. Class Distribution per Split

| Canonical Label | Train Count (%) | Val Count (%) | Test Count (%) | Combined Total (%) |
|---|---|---|---|---|
| `Facts` | 169,647 (15.11%) | 51,922 (17.7%) | 24,908 (16.62%) | **246,477 (15.74%)** |
| `Issue` | 12,769 (1.14%) | 4,257 (1.45%) | 1,836 (1.23%) | **18,862 (1.2%)** |
| `Arguments of Petitioner` | 64,985 (5.79%) | 24,707 (8.42%) | 14,520 (9.69%) | **104,212 (6.66%)** |
| `Arguments of Respondent` | 50,088 (4.46%) | 16,020 (5.46%) | 9,579 (6.39%) | **75,687 (4.83%)** |
| `Reasoning` | 202,328 (18.03%) | 67,112 (22.88%) | 36,688 (24.48%) | **306,128 (19.55%)** |
| `Decision` | 19,571 (1.74%) | 7,634 (2.6%) | 3,841 (2.56%) | **31,046 (1.98%)** |
| `None` | 602,988 (53.72%) | 121,710 (41.49%) | 58,496 (39.03%) | **783,194 (50.02%)** |