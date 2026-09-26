"""
predict.py — Test-time inference: blocking → features → scoring → thresholding.

Produces the final predictions dict {s1_id: set(matched_ids)} and the
candidate dict {s1_id: list(candidate_ids)} needed for submission files.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib

from data_loader import load_all_test
from preprocessing import preprocess_dataframe
from blocking import generate_candidates
from features import extract_features_batch, df_to_record_dict


def predict_pipeline(
    dataset_dir: str,
    model_dir: str,
    threshold: float,
    max_df_ratio: float = 0.005,
    max_candidates: int = 50,
):
    """Run the full test-time prediction pipeline with chunked scoring for memory safety."""
    t0 = time.time()

    # ---- Load ----
    s1, s2, s3 = load_all_test(dataset_dir)
    all_s1_ids = list(s1["entity_id"])

    # ---- Preprocess ----
    print("\nPreprocessing test data...", flush=True)
    s1 = preprocess_dataframe(s1)
    s2 = preprocess_dataframe(s2)
    s3 = preprocess_dataframe(s3)
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    print(f"  Test S2+S3 combined: {len(s2s3):,} records", flush=True)

    # ---- Blocking ----
    print("\nBlocking (test)...", flush=True)
    candidates = generate_candidates(s1, s2s3, max_df_ratio, max_candidates)

    total_pairs = sum(len(v) for v in candidates.values())
    print(f"  Test candidates: {total_pairs:,} pairs across {len(candidates):,} S1 entities", flush=True)

    # ---- Load model ----
    print("\nLoading model...", flush=True)
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    print(f"  Model loaded. Threshold = {threshold}", flush=True)

    # ---- Build record dictionaries ----
    print("\nIndexing record dictionaries...", flush=True)
    s1_records = df_to_record_dict(s1)
    s2s3_records = df_to_record_dict(s2s3)

    pairs = []
    for s1_id, cands in candidates.items():
        for c_id in cands:
            pairs.append((s1_id, c_id))
    print(f"  Total candidate pairs to score: {len(pairs):,}", flush=True)

    predictions = {sid: set() for sid in all_s1_ids}
    if len(pairs) == 0:
        elapsed = time.time() - t0
        print(f"\nPrediction complete in {elapsed:.0f}s (no candidates)", flush=True)
        return predictions, candidates, all_s1_ids

    # ---- Chunked Feature Extraction & Scoring ----
    print("\nScoring in memory-safe chunks...", flush=True)
    batch_size = 500_000
    n_batches = (len(pairs) + batch_size - 1) // batch_size

    for b in range(n_batches):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, len(pairs))
        batch_pairs = pairs[start_idx:end_idx]

        X_batch = extract_features_batch(s1_records, s2s3_records, batch_pairs)
        X_batch_scaled = scaler.transform(X_batch)
        probs_batch = model.predict_proba(X_batch_scaled)[:, 1]

        for (s1_id, s2_id), prob in zip(batch_pairs, probs_batch):
            if prob >= threshold:
                predictions[s1_id].add(s2_id)

        print(f"    Batch {b + 1}/{n_batches} scored ({end_idx:,}/{len(pairs):,} pairs)", flush=True)

    n_matched = sum(1 for v in predictions.values() if v)
    n_empty = sum(1 for v in predictions.values() if not v)
    total_links = sum(len(v) for v in predictions.values())
    print(f"\nPredictions: {n_matched:,} S1 with matches, "
          f"{n_empty:,} S1 empty, {total_links:,} total links", flush=True)

    elapsed = time.time() - t0
    print(f"Prediction pipeline complete in {elapsed:.0f}s", flush=True)
    return predictions, candidates, all_s1_ids
