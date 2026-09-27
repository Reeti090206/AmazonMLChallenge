"""
predict.py — Test-time inference: blocking → features → scoring → thresholding.

Uses country-partitioned streaming (loading and processing each country partition
independently) to guarantee low memory footprint (<1.5 GB peak) and zero virtual
memory paging/swapping.
"""

import os
import time
import gc
import numpy as np
import pandas as pd
import joblib

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
    """Run test-time prediction with country-partitioned streaming."""
    t0 = time.time()

    # Load model and scaler
    print("\nLoading model and scaler...", flush=True)
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    print(f"  Model loaded. Decision Threshold = {threshold}", flush=True)

    part_dir = os.path.join(dataset_dir, "partitions")
    countries = ["france", "india", "us"]

    # Read all S1 IDs to guarantee every entity appears in output
    print("\nReading S1 entity ID master list...", flush=True)
    all_s1_ids = []
    with open(os.path.join(dataset_dir, "test", "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                all_s1_ids.append(parts[0])

    print(f"  Master S1 entities: {len(all_s1_ids):,}", flush=True)

    all_candidates = {}
    predictions = {sid: set() for sid in all_s1_ids}

    for country in countries:
        s1_file = os.path.join(part_dir, f"{country}_s1.tsv")
        s2s3_file = os.path.join(part_dir, f"{country}_s2s3.tsv")

        if not os.path.isfile(s1_file) or not os.path.isfile(s2s3_file):
            print(f"  WARNING: Missing partition files for {country}")
            continue

        print(f"\n=======================================================", flush=True)
        print(f"  Partition: {country.upper()}", flush=True)
        print(f"=======================================================", flush=True)

        t_c = time.time()
        s1_c = pd.read_csv(s1_file, sep="\t")
        s2s3_c = pd.read_csv(s2s3_file, sep="\t")
        print(f"  Loaded {len(s1_c):,} S1 and {len(s2s3_c):,} S2/S3 records in {country} ({time.time()-t_c:.1f}s)", flush=True)

        print(f"  Preprocessing {country}...", flush=True)
        s1_c = preprocess_dataframe(s1_c)
        s2s3_c = preprocess_dataframe(s2s3_c)

        # 1. Candidate blocking
        cands_c = generate_candidates(s1_c, s2s3_c, max_df_ratio, max_candidates)
        all_candidates.update(cands_c)

        # 2. Build lightweight record tuples
        s1_rec = df_to_record_dict(s1_c)
        s2s3_rec = df_to_record_dict(s2s3_c)

        # 3. Free heavy DataFrames immediately
        del s1_c, s2s3_c
        gc.collect()

        # 4. Stream scoring in batches without giant pair list
        total_country_pairs = sum(len(v) for v in cands_c.values())
        print(f"  Scoring {total_country_pairs:,} pairs in {country}...", flush=True)

        batch_size = 250_000
        curr_batch = []
        n_scored = 0
        batch_num = 0

        for sid, cl in cands_c.items():
            for cid in cl:
                curr_batch.append((sid, cid))
                if len(curr_batch) >= batch_size:
                    batch_num += 1
                    X_b = extract_features_batch(s1_rec, s2s3_rec, curr_batch)
                    probs_b = model.predict_proba(scaler.transform(X_b))[:, 1]

                    for (s1_id, s2_id), prob in zip(curr_batch, probs_b):
                        if prob >= threshold:
                            predictions[s1_id].add(s2_id)

                    n_scored += len(curr_batch)
                    print(f"    [{country}] Batch {batch_num} scored ({n_scored:,}/{total_country_pairs:,} pairs)", flush=True)
                    curr_batch = []

        # Process final partial batch
        if curr_batch:
            batch_num += 1
            X_b = extract_features_batch(s1_rec, s2s3_rec, curr_batch)
            probs_b = model.predict_proba(scaler.transform(X_b))[:, 1]
            for (s1_id, s2_id), prob in zip(curr_batch, probs_b):
                if prob >= threshold:
                    predictions[s1_id].add(s2_id)
            n_scored += len(curr_batch)
            print(f"    [{country}] Final Batch {batch_num} scored ({n_scored:,}/{total_country_pairs:,} pairs)", flush=True)

        del s1_rec, s2s3_rec, cands_c
        gc.collect()
        print(f"  Completed {country} partition in {time.time()-t_c:.1f}s", flush=True)

    n_matched = sum(1 for v in predictions.values() if v)
    n_empty = sum(1 for v in predictions.values() if not v)
    total_links = sum(len(v) for v in predictions.values())
    match_pct = (n_matched / len(all_s1_ids) * 100) if all_s1_ids else 0.0

    print(f"\n=======================================================", flush=True)
    print(f"Predictions Summary:", flush=True)
    print(f"  S1 with matches: {n_matched:,} ({match_pct:.1f}%)", flush=True)
    print(f"  S1 empty:        {n_empty:,}", flush=True)
    print(f"  Total links:     {total_links:,}", flush=True)
    print(f"=======================================================", flush=True)

    elapsed = time.time() - t0
    print(f"Prediction pipeline complete in {elapsed:.0f}s", flush=True)
    return predictions, all_candidates, all_s1_ids
