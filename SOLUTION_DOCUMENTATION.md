# Amazon ML Challenge 2026: Business Entity Resolution
## Complete Solution Documentation & Verification Report

**Team Name:** TechForge  
**Status:** Complete & Officially Validated  
**Validation Exit Code:** 0 (PASS)  

---

### Key Milestone Achievements

| Milestone | Metric / Value | Details |
|---|---|---|
| **Test Dataset Ingested** | 11,708,212 records | S1: 1,732,544 \| S2: 4,892,109 \| S3: 5,083,559 |
| **Open-Set Coverage** | 3 Countries (`IN`, `US`, `FR`) | France ($N=259,452$) handled zero-shot |
| **Candidate Blocking** | 41,967,443 pairs | Rarity-ranked token inverted index ($K=25$) |
| **Pairs Scored** | 41,967,443 / 41,967,443 (100%) | 84 batches of 500,000 pairs completed |
| **Peak RAM Usage** | 8.28 GB (stabilized at 7.76 GB) | Memory-safe chunked inference |
| **Matched Links** | 1,944,807 entity links | 683,481 S1 entities with matches (39.4%) |
| **Unmatched S1 Entities** | 1,049,063 entities | 60.6% correctly predicted empty |
| **Validation Status** | **100% PASS (Exit Code 0)** | 0 orphan matches, exact 1,732,544 rows in both files |
| **Submission Package** | `TechForge_submission.zip` | 240.6 MB, exact required directory structure |

---

### Final Submission Package Structure
```
TechForge_submission.zip
├── output/
│   ├── matching_results.tsv      (1,732,544 rows, 50.2 MB)
│   └── candidate_pairs.tsv       (1,732,544 rows, 565.0 MB)
└── code/
    └── business_entity_resolution/
        ├── README.md
        ├── requirements.txt
        └── src/
            ├── __init__.py
            ├── blocking.py
            ├── data_loader.py
            ├── evaluate.py
            ├── features.py
            ├── predict.py
            ├── preprocessing.py
            ├── submission.py
            └── train.py
```
