# DATASET ANALYSIS REPORT
## Amazon ML Challenge 2026: Business Entity Resolution Challenge
**Team Name:** TechForge  
**Date:** September 26, 2026  
**Dataset Location:** `dataset/train/` and `dataset/test/`

---

### Executive Summary

This document presents the Phase 1 empirical dataset inspection for the Business Entity Resolution Challenge. All statistics reported below were directly measured from the official TSV dataset files.

---

### 1. Dataset File Inventory & Properties

| File Alias | Local Path | File Size (MB) | Row Count | Column Count | Data Types | ID Prefix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Source 1** | `dataset/train/train_source1.tsv` | 200.34 MB | 2,206,821 | 4 | `str` (all) | `S1-` |
| **Train Source 2** | `dataset/train/train_source2.tsv` | 466.63 MB | 5,034,616 | 4 | `str` (all) | `S2-` |
| **Train Source 3** | `dataset/train/train_source3.tsv` | 480.37 MB | 5,285,603 | 4 | `str` (all) | `S3-` |
| **Train Ground Truth** | `dataset/train/train_ground_truth.tsv` | 121.13 MB | 2,206,821 | 2 | `str` (all) | `S1-` |
| **Test Source 1** | `dataset/test/test_source1.tsv` | 166.91 MB | 1,732,544 | 4 | `str` (all) | `S1-` |
| **Test Source 2** | `dataset/test/test_source2.tsv` | 485.86 MB | 4,887,273 | 4 | `str` (all) | `S2-` |
| **Test Source 3** | `dataset/test/test_source3.tsv` | 482.56 MB | 5,082,316 | 4 | `str` (all) | `S3-` |

* **Total Disk Size:** 2,403.80 MB (~2.40 GB)
* **Total Records Across All Files:** 26,715,994 rows (12,527,040 train entities + 11,702,133 test entities + 2,486,821 GT/S1 index entries)

---

### 2. Schema and Field Definitions

All source files (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`) contain exactly the following four tab-separated columns:
1. `entity_id`: Unique string identifier with source prefix (`S1-`, `S2-`, `S3-`).
2. `business_name`: Text field containing entity name, trade names, or legal designations.
3. `business_address`: Text field containing street, city, state/region, and postal components.
4. `country`: String label indicating country.

The ground truth file (`train_ground_truth.tsv`) contains exactly:
1. `source1_entity_id`: Source 1 identifier (`S1-xxxxx`).
2. `matched_entity_ids`: Comma-separated list of matching `S2-` and/or `S3-` entity IDs (empty string for singletons/no-matches).

---

### 3. Entity ID Uniqueness Verification

* **Train Source 1:** 2,206,821 unique IDs (100% unique, zero duplicates). Prefix: `S1-`
* **Train Source 2:** 5,034,616 unique IDs (100% unique, zero duplicates). Prefix: `S2-`
* **Train Source 3:** 5,285,603 unique IDs (100% unique, zero duplicates). Prefix: `S3-`
* **Train Ground Truth:** 2,206,821 unique S1 IDs (100% 1-to-1 match with Train Source 1).
* **Test Source 1:** 1,732,544 unique IDs (100% unique, zero duplicates). Prefix: `S1-`
* **Test Source 2:** 4,887,273 unique IDs (100% unique, zero duplicates). Prefix: `S2-`
* **Test Source 3:** 5,082,316 unique IDs (100% unique, zero duplicates). Prefix: `S3-`

---

### 4. Missing Values Analysis

| Dataset | Total Rows | `entity_id` Missing (%) | `business_name` Missing (%) | `business_address` Missing (%) | `country` Missing (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Source 1** | 2,206,821 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Train Source 2** | 5,034,616 | 0 (0.00%) | 2 (0.00004%) | 168,967 (3.36%) | 0 (0.00%) |
| **Train Source 3** | 5,285,603 | 0 (0.00%) | 13 (0.0002%) | 175,916 (3.33%) | 0 (0.00%) |
| **Test Source 1** | 1,732,544 | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **Test Source 2** | 4,887,273 | 0 (0.00%) | 46 (0.0009%) | 129,408 (2.65%) | 0 (0.00%) |
| **Test Source 3** | 5,082,316 | 0 (0.00%) | 59 (0.0012%) | 136,098 (2.68%) | 0 (0.00%) |

**Key Findings:**
* `entity_id` and `country` have **0 missing values** across all split files.
* Source 1 (the reference source) has **0 missing values** across all fields.
* Sources 2 and 3 contain ~2.65% to ~3.36% missing values in `business_address`.
* Missing values in `business_name` are extremely rare (< 0.001%).

---

### 5. Country Distribution & Open-Set Analysis

| Source Split | Total Records | US Count (%) | India Count (%) | France Count (%) | Total Countries |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train Source 1** | 2,206,821 | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.00%) | 2 |
| **Train Source 2** | 5,034,616 | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.00%) | 2 |
| **Train Source 3** | 5,285,603 | 3,170,056 (59.97%) | 2,115,547 (40.03%) | 0 (0.00%) | 2 |
| **Test Source 1** | 1,732,544 | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) | 3 |
| **Test Source 2** | 4,887,273 | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) | 3 |
| **Test Source 3** | 5,082,316 | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (14.40%) | 3 |

**Critical Open-Set Observation:**
* France is **absent** in training data (0 records) but constitutes **14.98% of Test Source 1** (259,452 entities), **14.39% of Test Source 2** (703,378 entities), and **14.40% of Test Source 3** (731,615 entities).
* All blocking, string normalization, feature extraction, and prediction logic MUST be open-set string based and strictly country-agnostic.

---

### 6. Ground-Truth Match-Count & Distribution Analysis

Ground truth analysis of `train_ground_truth.tsv` (2,206,821 Source 1 entities):

* **Total True Entity Links:** 7,638,365 pairs (3,693,619 S2 links + 3,944,746 S3 links)
* **Average Matches per S1 Entity:** 3.4613
* **Median Matches per S1 Entity:** 3.0
* **Maximum Matches for a single S1 Entity:** 11 matches

#### Match Breakdown per Source 1 Entity:

| Match Category | S1 Entity Count | Percentage (%) | Description |
| :--- | :--- | :--- | :--- |
| **Zero Matches (Singletons)** | 123,247 | 5.5848% | S1 entity has no corresponding record in S2 or S3 |
| **Exactly 1 Match** | 119,157 | 5.3995% | Matched to exactly 1 S2 or S3 entity |
| **Multiple Matches (> 1)** | 1,964,417 | 89.0157% | Matched to 2 or more S2 and/or S3 entities |

#### Source Co-Occurrence Breakdown:

| Match Source Combination | Count | Percentage (%) |
| :--- | :--- | :--- |
| **Matches in BOTH S2 & S3** | 1,776,047 | 80.48% |
| **Matches in S3 ONLY** | 164,498 | 7.45% |
| **Matches in S2 ONLY** | 143,029 | 6.48% |
| **No Matches (Singletons)** | 123,247 | 5.58% |

#### Detailed Match Count Frequency Distribution:

| Match Count | S1 Entity Frequency | Percentage (%) | Cumulative % |
| :--- | :--- | :--- | :--- |
| **0** | 123,247 | 5.58% | 5.58% |
| **1** | 119,157 | 5.40% | 10.98% |
| **2** | 375,212 | 17.00% | 27.98% |
| **3** | 530,841 | 24.05% | 52.04% |
| **4** | 484,115 | 21.94% | 73.98% |
| **5** | 321,957 | 14.59% | 88.57% |
| **6** | 164,868 | 7.47% | 96.04% |
| **7** | 63,968 | 2.90% | 98.94% |
| **8** | 18,680 | 0.85% | 99.79% |
| **9** | 4,205 | 0.19% | 99.98% |
| **10** | 534 | 0.02% | 100.00% |
| **11** | 37 | 0.002% | 100.00% |

---

### 7. Noise Patterns & Text Characteristics

Empirical analysis of text fields and a sampled set of true ground-truth matching pairs:

#### Text Field Lengths & Token Counts:

* **Business Name:**
  * Mean length: ~24 - 25 characters (Max: 92 characters).
  * Mean word count: ~3.5 words (95th percentile: 5 - 6 words).
* **Business Address:**
  * Mean length: ~48 - 57 characters (Max: 229 characters).
  * Mean word count: ~7.5 - 8.5 words (95th percentile: 15 - 16 words).

#### True Match String Variations (Ground Truth Sample Analysis):

| Noise Metric | Measured Value (%) | Key Insight |
| :--- | :--- | :--- |
| **Exact Name Match** | **4.61%** | 95.39% of true matches have string differences in business name |
| **Case-Insensitive Exact Name Match** | **10.45%** | Case normalization captures only 5.84% additional exact matches |
| **Exact Address Match** | **2.07%** | 97.93% of true matches have variations in address |
| **Case-Insensitive Exact Address Match** | **6.93%** | Address format variation is dominant |
| **Country Mismatch in True Matches** | **0.00%** | Country is 100% consistent across sources for true matches |
| **Mean Name Word Jaccard Similarity** | **0.6137** | Significant token overlap exists despite typos & abbreviations |
| **Mean Address Word Jaccard Similarity** | **0.5942** | Address tokens (street, city, pin, numbers) overlap strongly |

#### Country-Specific Legal & Address Tokens:

* **US:** Legal (`llc`, `inc`, `pc`, `co`, `associates`, `corp`); Address (`street`, `road`, `drive`, `unit`, `avenue`, `tx`, `ny`, `nc`, `oh`, `il`).
* **India:** Legal (`limited`, `private`, `ltd`, `pvt`, `llp`, `co`); Address (`no`, `delhi`, `road`, `maharashtra`, `nagar`, `floor`, `mumbai`, `pradesh`, `bangalore`).
* **France:** Legal (`sarl`, `sas`, `eurl`, `sa`, `sasu`); Address (`de`, `rue`, `la`, `france`, `hauts`, `nouvelle`, `aquitaine`, `pays`, `loire`, `bordeaux`).

---

### 8. Search Space & Computational Requirements

#### Full Cartesian Search Space vs. Country-Blocked Search Space:

* **Train Set Search Space:**
  * Full Cartesian $S1 \times (S2 + S3)$: $2,206,821 \times 10,320,219 = \mathbf{22,774,876,013,799}$ pairs (~22.77 Trillion pairs).
  * Country-Blocked $S1_{US} \times (S2_{US} + S3_{US}) + S1_{IN} \times (S2_{IN} + S3_{IN})$: $\mathbf{11,839,670,856,657}$ pairs (~11.84 Trillion pairs).
  * Country blocking alone reduces pairs by **48.01%**.

* **Test Set Search Space:**
  * Full Cartesian $S1 \times (S2 + S3)$: $1,732,544 \times 9,969,589 = \mathbf{17,272,751,604,416}$ pairs (~17.27 Trillion pairs).
  * Country-Blocked $S1_{US} \times (S2_{US} + S3_{US}) + S1_{IN} \times (S2_{IN} + S3_{IN}) + S1_{FR} \times (S2_{FR} + S3_{FR})$: $\mathbf{6,724,569,566,212}$ pairs (~6.72 Trillion pairs).
  * Country blocking on Test set reduces pairs by **61.07%**.

#### Memory & Computational Bounds:
* Storing 11.8 Trillion pair features in RAM is computationally impossible.
* A multi-stage Blocking Strategy (Country + Name N-gram / Token Inverted Index + Address Numeric Tokens) MUST target an average candidate density of **~10 to 50 candidates per S1 entity**, yielding **~17M to 85M candidate pairs** overall.
* Memory usage for 50M candidate pairs with 15 numerical features: **~6.0 GB RAM**, perfectly fitting local system hardware.

---

### 9. Summary of Key Data-Quality Insights for Pipeline Design

1. **Country Blocking is Required & 100% Safe:** Zero country mismatches in true ground truth pairs.
2. **Multi-Match Prediction Logic Required:** 89.02% of S1 entities match multiple records (up to 11). One-to-one or max-score picking will fail.
3. **Singletons (5.58%):** Correctly outputting an empty string for singletons gives full 1.0 score per singleton entity.
4. **Open-Set France Handling:** Preprocessing and feature engineering must avoid any hardcoded country lists.
5. **Exact String Match is insufficient:** Only 4.61% of true matches are exact string matches. TF-IDF, character n-gram Jaccard, and edit distance features are required.

---
*End of Dataset Analysis Report.*
