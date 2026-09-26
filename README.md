# Amazon ML Challenge 2026: Business Entity Resolution Challenge

**Team:** TechForge  
**Competition:** Amazon ML Challenge 2026 — Business Entity Resolution  
**Status:** Completed & Officially Validated (Exit Code 0)  

---

## 1. Project Overview

This repository contains the complete end-to-end solution for the Amazon ML Challenge 2026 Business Entity Resolution Challenge. The objective is to resolve and link business records across three disparate data sources (`Source 1`, `Source 2`, and `Source 3`) into unified entities under the official competition macro-averaged $F_{0.5}$ metric.

### Key Highlights:
- **Scalability:** Successfully ingested and processed **11.7 million test records** across 3 sources, generating **41,967,443 candidate pairs** and resolving **1,944,807 entity links** across **1,732,544 Source 1 entities**.
- **Open-Set Generalization:** Handles unseen open-set countries (`France` with 259k entities in the test set) zero-shot using country-agnostic, language-neutral normalisation without hardcoded rules.
- **Rarity-Ranked Blocking:** Inverted index with document frequency thresholding ($DF \le 0.005$) and posting truncation ($\le 1,500$ entries), achieving 130,000 entities/sec retrieval speed.
- **15-Dimensional Feature Engineering:** Pairwise similarity signals combining Jaccard, character 3-gram Jaccard, C++ RapidFuzz token/partial Levenshtein ratios, address numeric overlaps, and length divergence metrics.
- **Macro $F_{0.5}$ Calibration:** Calibrated decision threshold ($\theta^* = 0.90$) prioritizing precision ($\beta = 0.5$) with 83.0% validation precision.
- **100% Validation Pass:** Fully verified by `validate_submission.py` with exit code 0, 100% row coverage (1,732,544 rows), and zero candidate superset violations.

---

## 2. Repository Structure

```
├── .gitignore
├── README.md                                  # Repository overview and instructions
├── run_pipeline.py                            # Root pipeline entry point wrapper
├── Documentation_template.md                  # Official solution documentation template
├── SOLUTION_DOCUMENTATION.md                  # Comprehensive solution report & metrics
├── DATASET_ANALYSIS.md                        # Phase 1 Exploratory Data Analysis report
│
├── utils/
│   └── validate_submission.py                 # Official submission validation script
│
├── models/
│   ├── model.pkl                              # Trained calibrated classifier
│   ├── scaler.pkl                             # Feature StandardScaler
│   └── train_results.json                     # Training metadata and metrics
│
└── code/
    └── business_entity_resolution/
        ├── README.md                          # Source code overview & run guide
        ├── requirements.txt                   # Minimal runtime dependencies
        └── src/
            ├── __init__.py
            ├── data_loader.py                 # TSV loading & schema validation
            ├── preprocessing.py               # Unicode normalization & tokenization
            ├── blocking.py                    # Rarity-ranked inverted index blocking
            ├── features.py                    # 15 pairwise similarity features
            ├── evaluate.py                    # Official macro F0.5 metric evaluation
            ├── train.py                       # Training & threshold calibration
            ├── predict.py                     # Memory-safe chunked batch inference
            └── submission.py                  # End-to-end pipeline orchestrator
```

---

## 3. Quick Start & Execution

### Installation:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### Run End-to-End Pipeline:
```bash
python run_pipeline.py --dataset-dir dataset --output-dir output --model-dir models --skip-train --threshold 0.9 --max-candidates 25
```

### Validate Submission Files:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 4. Final Submission Archive

The final competition archive `TechForge_submission.zip` (240.6 MB) adheres strictly to the required competition submission format:
```
TechForge_submission.zip
├── output/
│   ├── matching_results.tsv                   (1,732,544 rows, 50.2 MB)
│   └── candidate_pairs.tsv                    (1,732,544 rows, 565.0 MB)
└── code/
    └── business_entity_resolution/
        ├── README.md
        ├── requirements.txt
        └── src/
            └── [all source modules]
```
