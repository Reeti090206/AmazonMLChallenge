"""
submission.py — End-to-end pipeline entry point.

Orchestrates: training → prediction → output generation → validation.
Produces the two required output files:
    output/matching_results.tsv
    output/candidate_pairs.tsv

Usage (from the repository root):
    python run_pipeline.py \
        --dataset-dir dataset \
        --output-dir output \
        --model-dir models \
        [--sample-s1 100000]
"""

import argparse
import os
import sys
import subprocess

from train import train_pipeline
from predict import predict_pipeline


def write_matching_results(predictions: dict, all_s1_ids: list, output_path: str):
    """Write matching_results.tsv — the scored output file.

    Format: TAB-separated, columns source1_entity_id and matched_entity_ids.
    Every S1 entity in test_source1 appears exactly once.
    matched_entity_ids is a comma-separated string (empty for no matches).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(all_s1_ids):
            matched = predictions.get(s1_id, set())
            matched_str = ",".join(sorted(matched)) if matched else ""
            f.write(f"{s1_id}\t{matched_str}\n")
    print(f"  Wrote {output_path}: {len(all_s1_ids):,} rows", flush=True)


def write_candidate_pairs(candidates: dict, predictions: dict, all_s1_ids: list, output_path: str):
    """Write candidate_pairs.tsv — the final candidate set fed to the model.

    Format: TAB-separated, columns source1_entity_id and candidate_entity_ids.
    Every S1 entity appears exactly once.
    Ensures any predicted match is guaranteed to be in the candidate set.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(all_s1_ids):
            cands = set(candidates.get(s1_id, []))
            # Strict superset guarantee: matching results MUST be a subset of candidate pairs
            matched = predictions.get(s1_id, set())
            if matched:
                cands.update(matched)
            cands_str = ",".join(sorted(cands)) if cands else ""
            f.write(f"{s1_id}\t{cands_str}\n")
    print(f"  Wrote {output_path}: {len(all_s1_ids):,} rows", flush=True)


def run_validator(output_dir: str, dataset_dir: str, validator_path: str = None):
    """Run the official validate_submission.py script."""
    if validator_path is None:
        candidates_paths = [
            os.path.join("utils", "validate_submission.py"),
            os.path.join("student_resource", "utils", "validate_submission.py"),
        ]
        for p in candidates_paths:
            if os.path.isfile(p):
                validator_path = p
                break

    if validator_path is None or not os.path.isfile(validator_path):
        print("  WARNING: Official validator not found. Skipping validation.", flush=True)
        return None

    matching = os.path.join(output_dir, "matching_results.tsv")
    candidate = os.path.join(output_dir, "candidate_pairs.tsv")
    test_dir = os.path.join(dataset_dir, "test")

    cmd = [
        sys.executable, validator_path,
        "--matching", matching,
        "--candidate", candidate,
        "--test-dir", test_dir,
    ]
    print(f"  Running validator: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution — full pipeline"
    )
    parser.add_argument("--dataset-dir", default="dataset",
                        help="Path to the dataset root (default: dataset)")
    parser.add_argument("--output-dir", default="output",
                        help="Path for output files (default: output)")
    parser.add_argument("--model-dir", default="models",
                        help="Path for model artefacts (default: models)")
    parser.add_argument("--sample-s1", type=int, default=None,
                        help="Subsample N S1 entities for faster training")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Fraction of S1 entities for validation (default: 0.2)")
    parser.add_argument("--max-df-ratio", type=float, default=0.005,
                        help="Blocking token max document-frequency ratio (default: 0.005)")
    parser.add_argument("--max-candidates", type=int, default=50,
                        help="Max candidates per S1 entity (default: 50)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing model")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (skip auto-selection)")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("Business Entity Resolution Pipeline — TechForge", flush=True)
    print("=" * 60, flush=True)

    # ---- Phase 1: Training ----
    if not args.skip_train:
        print("\n>>> PHASE 1: Training", flush=True)
        train_results = train_pipeline(
            dataset_dir=args.dataset_dir,
            output_dir=args.model_dir,
            val_fraction=args.val_fraction,
            sample_s1=args.sample_s1,
            max_df_ratio=args.max_df_ratio,
            max_candidates=args.max_candidates,
        )
        threshold = args.threshold or train_results["best_threshold"]
    else:
        if args.threshold is None:
            raise ValueError("--threshold is required when --skip-train is set")
        threshold = args.threshold
        print(f"\n>>> Skipping training. Using threshold = {threshold}", flush=True)

    # ---- Phase 2: Test Prediction ----
    print("\n>>> PHASE 2: Test Prediction", flush=True)
    predictions, candidates, all_s1_ids = predict_pipeline(
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        threshold=threshold,
        max_df_ratio=args.max_df_ratio,
        max_candidates=args.max_candidates,
    )

    # ---- Phase 3: Write Output Files ----
    print("\n>>> PHASE 3: Writing Output Files", flush=True)
    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
    write_matching_results(predictions, all_s1_ids, matching_path)
    write_candidate_pairs(candidates, predictions, all_s1_ids, candidate_path)

    # Verify every match is in candidates
    n_orphan = 0
    for s1_id, matched in predictions.items():
        cand_set = set(candidates.get(s1_id, []))
        orphans = matched - cand_set
        n_orphan += len(orphans)
    if n_orphan > 0:
        print(f"  WARNING: {n_orphan} matched IDs are NOT in candidate_pairs!", flush=True)
    else:
        print("  [PASS] All matched IDs are present in candidate_pairs.tsv", flush=True)

    # ---- Phase 4: Validate ----
    print("\n>>> PHASE 4: Running Official Validator", flush=True)
    ret = run_validator(args.output_dir, args.dataset_dir)
    if ret == 0:
        print("  [PASS] Validation Successful (exit code 0)", flush=True)
    elif ret is not None:
        print(f"  [FAIL] Validation Failed (exit code {ret})", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Pipeline complete.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
