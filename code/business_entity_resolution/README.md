# Business Entity Resolution Pipeline

**Team:** TechForge  
**Challenge:** Amazon ML Challenge 2026 — Business Entity Resolution  

---

## 1. Overview

This package implements an end-to-end entity resolution pipeline designed to resolve business entity records across three distinct sources (`Source 1`, `Source 2`, and `Source 3`) into unified real-world entities under the macro-averaged $F_{0.5}$ metric.

### Pipeline Architecture:
1. **Schema Validation & Ingestion** (`data_loader.py`):
   - Fast, resilient TSV loading with tab-separation validation, missingness checks, and ID prefix assertion (`S1-`, `S2-`, `S3-`).
2. **Text Normalisation & Feature Extraction** (`preprocessing.py`):
   - Country-agnostic, open-set safe Unicode NFKD normalization, accent stripping, non-alphanumeric filtering, word tokenization, and numeric postal/building code extraction.
3. **Candidate Blocking & Indexing** (`blocking.py`):
   - Strict country-partitioned candidate generation with document-frequency filtering ($DF > 0.005$) to discard non-discriminative business stopwords.
   - Inverted name-token index with candidate ranking by shared token count, bounded to Top-$K$ candidates per Source 1 entity.
4. **Pairwise Feature Engineering** (`features.py`):
   - 15 country-agnostic pairwise string, character, and numeric similarity signals (Jaccard, 3-gram character Jaccard, RapidFuzz token/partial ratio, numeric token overlap, length difference ratios).
5. **Discriminative Classification & F0.5 Calibration** (`train.py`, `evaluate.py`):
   - Logistic regression / linear classifier with balanced class weighting.
   - Per-entity macro $F_{0.5}$ metric optimization on held-out validation split.
6. **Inference & Submission Formatting** (`predict.py`, `submission.py`):
   - Generates exact formatted `output/matching_results.tsv` and `output/candidate_pairs.tsv` satisfying all competition constraints, validated via `validate_submission.py`.

---

## 2. Directory Structure

```
code/
└── business_entity_resolution/
    ├── requirements.txt
    ├── README.md
    └── src/
        ├── __init__.py
        ├── data_loader.py
        ├── preprocessing.py
        ├── blocking.py
        ├── features.py
        ├── evaluate.py
        ├── train.py
        ├── predict.py
        └── submission.py
```

---

## 3. Installation

Install all required Python packages (Python 3.9+ recommended):

```bash
pip install -r requirements.txt
```

---

## 4. Execution

To run the complete end-to-end pipeline (training, validation, test inference, output generation, and validation):

```bash
python -m code.business_entity_resolution.src.submission \
    --dataset-dir dataset \
    --output-dir output \
    --model-dir models \
    --val-fraction 0.2 \
    --max-candidates 100
```

### Command-Line Arguments:
- `--dataset-dir`: Directory containing `train/` and `test/` TSV folders (default: `dataset`).
- `--output-dir`: Target directory for `matching_results.tsv` and `candidate_pairs.tsv` (default: `output`).
- `--model-dir`: Directory to store/load trained model artefacts (default: `models`).
- `--sample-s1`: Optional subsample of Source 1 entities for fast iteration.
- `--val-fraction`: Held-out validation split ratio (default: `0.2`).
- `--max-candidates`: Maximum candidate pairs per Source 1 entity (default: `100` or `200`).
- `--skip-train`: Skip training and evaluate using pre-existing saved model weights.
- `--threshold`: Specific probability threshold override.
