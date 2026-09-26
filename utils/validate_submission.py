"""
validate_submission.py — Official submission validator for Amazon ML Challenge 2026.

Validates that:
1. Both output files exist: matching_results.tsv and candidate_pairs.tsv.
2. Headers are exact:
   - matching_results.tsv: source1_entity_id\tmatched_entity_ids
   - candidate_pairs.tsv: source1_entity_id\tcandidate_entity_ids
3. Exactly one row per test Source 1 entity (1,732,544 entities), no duplicates, no missing.
4. Every predicted match in matching_results.tsv is a subset of candidate_pairs.tsv.
"""

import argparse
import os
import sys


def validate(matching_path: str, candidate_path: str, test_dir: str) -> int:
    errors = []

    # 1. Existence
    if not os.path.isfile(matching_path):
        errors.append(f"Matching file does not exist: {matching_path}")
    if not os.path.isfile(candidate_path):
        errors.append(f"Candidate file does not exist: {candidate_path}")
    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return 1

    # Load test source 1 IDs
    s1_test_path = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(s1_test_path):
        errors.append(f"test_source1.tsv not found in {test_dir}")
        return 1

    print("Loading test_source1.tsv IDs...")
    expected_s1_ids = set()
    with open(s1_test_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        eid_idx = header.index("entity_id") if "entity_id" in header else 0
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[eid_idx]:
                expected_s1_ids.add(parts[eid_idx])
    n_expected = len(expected_s1_ids)
    print(f"  Expected Source 1 entities: {n_expected:,}")

    # Validate matching_results.tsv
    print("\nValidating matching_results.tsv...")
    matching_dict = {}
    with open(matching_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "source1_entity_id\tmatched_entity_ids":
            errors.append(f"Invalid header in matching_results.tsv: '{header}'")
        for line_num, line in enumerate(f, start=2):
            parts = line.strip("\r\n").split("\t")
            if len(parts) != 2:
                errors.append(f"matching_results.tsv line {line_num}: expected 2 tab-separated columns, got {len(parts)}")
                break
            sid, matched_str = parts[0], parts[1]
            if sid in matching_dict:
                errors.append(f"Duplicate entity_id in matching_results.tsv: {sid}")
                break
            matched_set = set(matched_str.split(",")) if matched_str else set()
            matching_dict[sid] = matched_set

    if len(matching_dict) != n_expected:
        errors.append(f"matching_results.tsv row count ({len(matching_dict):,}) != expected ({n_expected:,})")

    missing_s1_m = expected_s1_ids - set(matching_dict.keys())
    if missing_s1_m:
        errors.append(f"matching_results.tsv is missing {len(missing_s1_m):,} test entities")

    matched_count = sum(len(v) for v in matching_dict.values())
    entities_with_matches = sum(1 for v in matching_dict.values() if v)
    print(f"  Rows: {len(matching_dict):,}")
    print(f"  Entities with >= 1 match: {entities_with_matches:,} ({entities_with_matches/n_expected*100:.1f}%)")
    print(f"  Total matched links: {matched_count:,}")

    # Validate candidate_pairs.tsv
    print("\nValidating candidate_pairs.tsv and subset invariant...")
    candidate_s1_seen = set()
    orphan_matches = 0
    total_candidates = 0

    with open(candidate_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "source1_entity_id\tcandidate_entity_ids":
            errors.append(f"Invalid header in candidate_pairs.tsv: '{header}'")
        for line_num, line in enumerate(f, start=2):
            parts = line.strip("\r\n").split("\t")
            if len(parts) != 2:
                errors.append(f"candidate_pairs.tsv line {line_num}: expected 2 tab-separated columns, got {len(parts)}")
                break
            sid, cands_str = parts[0], parts[1]
            if sid in candidate_s1_seen:
                errors.append(f"Duplicate entity_id in candidate_pairs.tsv: {sid}")
                break
            candidate_s1_seen.add(sid)
            cands_set = set(cands_str.split(",")) if cands_str else set()
            total_candidates += len(cands_set)

            # Check subset invariant: matching_results[sid] <= candidate_pairs[sid]
            matched = matching_dict.get(sid, set())
            diff = matched - cands_set
            if diff:
                orphan_matches += len(diff)

    if len(candidate_s1_seen) != n_expected:
        errors.append(f"candidate_pairs.tsv row count ({len(candidate_s1_seen):,}) != expected ({n_expected:,})")

    missing_s1_c = expected_s1_ids - candidate_s1_seen
    if missing_s1_c:
        errors.append(f"candidate_pairs.tsv is missing {len(missing_s1_c):,} test entities")

    if orphan_matches > 0:
        errors.append(f"Subset violation: {orphan_matches} matches are NOT present in candidate_pairs!")

    print(f"  Rows: {len(candidate_s1_seen):,}")
    print(f"  Total candidate pairs: {total_candidates:,}")
    print(f"  Subset violation count: {orphan_matches}")

    # Report results
    print("\n" + "=" * 60)
    if errors:
        print("[FAIL] Validation failed with errors:")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("[PASS] ALL VALIDATION CHECKS PASSED SUCCESSFULLY (Exit Code 0)")
        print("  - Schema: Exact TSV columns verified")
        print("  - Coverage: 100% of test Source 1 entities represented (1,732,544 rows)")
        print("  - Consistency: No duplicate IDs, no missing rows")
        print("  - Superset Invariant: 100% of matches are contained in candidate pairs")
        print("=" * 60)
        return 0


def main():
    parser = argparse.ArgumentParser(description="Submission validator")
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="dataset/test")
    args = parser.parse_args()

    ret = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(ret)


if __name__ == "__main__":
    main()
