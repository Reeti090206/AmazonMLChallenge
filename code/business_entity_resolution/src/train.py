"""
train.py — Model training, threshold selection, and validation.

Trains a Logistic Regression classifier on labelled candidate pairs
derived from the training ground truth. Optimises the decision
threshold for macro F0.5 on a held-out validation split.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_loader import load_all_train, parse_matched_ids
from preprocessing import preprocess_dataframe
from blocking import generate_candidates, measure_blocking_recall
from features import (
    FEATURE_NAMES, extract_features_batch, df_to_record_dict,
)
from evaluate import evaluate_predictions


def _build_ground_truth_dict(gt_df: pd.DataFrame) -> dict:
    """Convert GT DataFrame to {s1_id: set(matched_ids)}."""
    gt = {}
    for s1_id, matched_str in zip(gt_df["source1_entity_id"].values, gt_df["matched_entity_ids"].values):
        if matched_str and not pd.isna(matched_str):
            gt[s1_id] = set(str(matched_str).split(","))
        else:
            gt[s1_id] = set()
    return gt


def _label_candidates(candidates: dict, gt_dict: dict):
    """Create labelled (s1_id, s2s3_id, label) triples from candidates + GT."""
    pairs = []
    labels = []
    for s1_id, cands in candidates.items():
        true_set = gt_dict.get(s1_id, set())
        for c_id in cands:
            pairs.append((s1_id, c_id))
            labels.append(1 if c_id in true_set else 0)
    return pairs, np.array(labels, dtype=np.int8)


def train_pipeline(
    dataset_dir: str,
    output_dir: str,
    val_fraction: float = 0.2,
    sample_s1: int = None,
    max_df_ratio: float = 0.005,
    max_candidates: int = 50,
    random_seed: int = 42,
):
    """Full training pipeline: load -> preprocess -> block -> featurize -> train -> validate."""
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(random_seed)
    results = {}

    # ---- Load ----
    t0 = time.time()
    s1, s2, s3, gt = load_all_train(dataset_dir)

    # Subsample S1 upfront if requested for rapid iteration
    if sample_s1 is not None and sample_s1 < len(s1):
        s1 = s1.sample(n=sample_s1, random_state=random_seed).reset_index(drop=True)
        print(f"  Subsampled to {sample_s1:,} S1 entities for training", flush=True)

    # ---- Preprocess ----
    print("\nPreprocessing...", flush=True)
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    print(f"  S2+S3 combined: {len(s2s3):,} records", flush=True)

    # ---- Split S1 into train / val ----
    all_s1_ids = np.array(s1["entity_id"].values, dtype=str)
    np.random.shuffle(all_s1_ids)

    n_val = int(len(all_s1_ids) * val_fraction)
    val_ids = set(all_s1_ids[:n_val])
    train_ids = set(all_s1_ids[n_val:])
    print(f"  Train S1: {len(train_ids):,}, Val S1: {len(val_ids):,}", flush=True)

    # ---- GT dict ----
    print("Building ground truth lookup...", flush=True)
    gt_dict = _build_ground_truth_dict(gt)

    # ---- Blocking on train S1 ----
    print("\nBlocking (train split)...", flush=True)
    s1_train = s1[s1["entity_id"].isin(train_ids)]
    train_cands = generate_candidates(s1_train, s2s3, max_df_ratio, max_candidates)

    gt_train = gt[gt["source1_entity_id"].isin(train_ids)]
    br_train = measure_blocking_recall(train_cands, gt_train)
    print(f"  Train blocking recall: {br_train['blocking_recall']:.4f} "
          f"({br_train['found_pairs']:,}/{br_train['total_true_pairs']:,})", flush=True)
    results["train_blocking_recall"] = br_train

    # ---- Blocking on val S1 ----
    print("\nBlocking (val split)...", flush=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)]
    val_cands = generate_candidates(s1_val, s2s3, max_df_ratio, max_candidates)

    gt_val = gt[gt["source1_entity_id"].isin(val_ids)]
    br_val = measure_blocking_recall(val_cands, gt_val)
    print(f"  Val blocking recall: {br_val['blocking_recall']:.4f} "
          f"({br_val['found_pairs']:,}/{br_val['total_true_pairs']:,})", flush=True)
    results["val_blocking_recall"] = br_val

    # ---- Feature extraction (train) ----
    print("\nExtracting features (train)...", flush=True)
    s1_records = df_to_record_dict(s1)
    s2s3_records = df_to_record_dict(s2s3)

    train_pairs, train_labels = _label_candidates(train_cands, gt_dict)
    print(f"  Train pairs: {len(train_pairs):,} "
          f"(pos={train_labels.sum():,}, neg={len(train_labels) - train_labels.sum():,})", flush=True)

    X_train = extract_features_batch(s1_records, s2s3_records, train_pairs)
    print(f"  Feature matrix shape: {X_train.shape}", flush=True)

    # ---- Train model ----
    print("\nTraining model...", flush=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced",
        random_state=random_seed, n_jobs=-1,
    )
    model.fit(X_train_scaled, train_labels)
    print("  Model trained.", flush=True)

    coef = dict(zip(FEATURE_NAMES, model.coef_[0]))
    print("  Feature coefficients:", flush=True)
    for name, c in sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:30s} {c:+.4f}", flush=True)
    results["feature_coefficients"] = coef

    # ---- Feature extraction (val) ----
    print("\nExtracting features (val)...", flush=True)
    val_pairs, val_labels = _label_candidates(val_cands, gt_dict)
    print(f"  Val pairs: {len(val_pairs):,} "
          f"(pos={val_labels.sum():,}, neg={len(val_labels) - val_labels.sum():,})", flush=True)

    X_val = extract_features_batch(s1_records, s2s3_records, val_pairs)

    # ---- Threshold optimisation ----
    print("\nOptimising threshold for F0.5...", flush=True)
    X_val_scaled = scaler.transform(X_val)
    val_probs = model.predict_proba(X_val_scaled)[:, 1]

    gt_val_dict = {sid: gt_dict.get(sid, set()) for sid in val_ids}
    best_threshold = 0.5
    best_f05 = 0.0
    threshold_results = []

    for thr in np.arange(0.1, 0.95, 0.05):
        preds = {sid: set() for sid in val_ids}
        for i, (s1_id, s2_id) in enumerate(val_pairs):
            if val_probs[i] >= thr:
                preds[s1_id].add(s2_id)
        metrics = evaluate_predictions(preds, gt_val_dict)
        threshold_results.append({"threshold": round(float(thr), 2), **metrics})
        print(f"    thr={thr:.2f}: F0.5={metrics['macro_f05']:.4f}, "
              f"P={metrics['macro_precision']:.4f}, R={metrics['macro_recall']:.4f}", flush=True)
        if metrics["macro_f05"] > best_f05:
            best_f05 = metrics["macro_f05"]
            best_threshold = round(float(thr), 2)

    print(f"\n  Best threshold: {best_threshold}  (F0.5 = {best_f05:.4f})", flush=True)
    results["best_threshold"] = best_threshold
    results["best_val_f05"] = best_f05
    results["threshold_sweep"] = threshold_results

    # ---- Save model ----
    model_path = os.path.join(output_dir, "model.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n  Model saved to {model_path}", flush=True)
    results["model_path"] = model_path
    results["scaler_path"] = scaler_path

    # ---- Save experiment log ----
    log_path = os.path.join(output_dir, "train_results.json")
    json_safe = {}
    for k, v in results.items():
        if isinstance(v, dict):
            json_safe[k] = {
                kk: (list(vv) if isinstance(vv, set) else vv)
                for kk, vv in v.items()
            }
        else:
            json_safe[k] = v
    with open(log_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nTraining pipeline complete in {elapsed:.0f}s", flush=True)
    return results
