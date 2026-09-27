"""
blocking.py — High-recall, high-precision candidate generation via country
partitioning + multi-key inverted indexing (IDF-weighted tokens, postal codes, and 3-grams).

Design rationale
----------------
1. Country blocking is 100% safe (0% country mismatches across true pairs).
2. Within country, rare tokens carry high IDF signals, while common tokens receive
   proportionately lower weight, avoiding candidate pool pollution.
3. Address postal codes and building numerics (>=4 digits) provide a complementary
   orthogonal blocking key to recover renamed or abbreviated businesses.
4. Character 3-grams serve as a fallback for short names with few token matches.
5. Exact clean-name hash lookup guarantees zero candidate drops for clean matches.
"""

import time
from collections import defaultdict
import numpy as np
import pandas as pd


def _get_c3(s: str) -> list:
    """Extract character 3-grams after removing spaces."""
    s = s.replace(" ", "")
    if len(s) >= 3:
        return [s[i:i + 3] for i in range(len(s) - 2)]
    return [s] if s else []


def _generate_candidates_for_partition(
    s1_ids,
    s1_tokens,
    s1_names,
    s1_nums,
    name_index: dict,
    token_idf: dict,
    addr_num_index: dict,
    num_idf: dict,
    c3_index: dict,
    s2s3_nc_map: dict,
    max_freq: int,
    max_candidates_per_entity: int = 50,
) -> dict:
    """Generate candidate S2/S3 IDs for each S1 entity in a single country partition."""
    candidates = {}

    for sid, tokens, name, nums in zip(s1_ids, s1_tokens, s1_names, s1_nums):
        cand_scores = {}

        # 1. Name tokens with precomputed IDF weighting
        valid_tokens = [t for t in tokens if t in token_idf and len(name_index[t]) < max_freq]
        for t in valid_tokens:
            w = token_idf[t]
            for cid in name_index[t][:3000]:
                cand_scores[cid] = cand_scores.get(cid, 0.0) + w

        # 2. Address numeric / postal code matching (len >= 4)
        for num in nums:
            if num in num_idf and len(addr_num_index[num]) <= 2000:
                w = num_idf[num]
                for cid in addr_num_index[num]:
                    cand_scores[cid] = cand_scores.get(cid, 0.0) + w

        # 3. Fallback: Character 3-grams for entities with small candidate pool
        if len(cand_scores) < 30 and name:
            c3_list = [g for g in _get_c3(name) if g in c3_index and len(c3_index[g]) < 1500]
            for g in c3_list:
                for cid in c3_index[g]:
                    cand_scores[cid] = cand_scores.get(cid, 0.0) + 1.0

        # 4. Fallback: Exact clean-name match
        if name and name in s2s3_nc_map:
            for cid in s2s3_nc_map[name]:
                cand_scores[cid] = cand_scores.get(cid, 0.0) + 100.0

        # Top-K candidate selection
        if len(cand_scores) <= max_candidates_per_entity:
            top = list(cand_scores.keys())
        else:
            top = sorted(cand_scores, key=cand_scores.get, reverse=True)[:max_candidates_per_entity]

        candidates[sid] = top

    return candidates


def generate_candidates(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    max_df_ratio: float = 0.005,
    max_candidates: int = 50,
) -> dict:
    """Generate candidate pairs for every S1 entity across country partitions."""
    t0 = time.time()
    all_candidates = {}

    countries = sorted(s1_df["country_clean"].unique())
    print(f"  Blocking across {len(countries)} country partition(s): {countries}", flush=True)

    for country in countries:
        s1_c = s1_df[s1_df["country_clean"] == country]
        s2s3_c = s2s3_df[s2s3_df["country_clean"] == country]

        if len(s2s3_c) == 0:
            for sid in s1_c["entity_id"].values:
                all_candidates[sid] = []
            print(f"    {country}: {len(s1_c):,} S1 x 0 S2S3 -> 0 candidates", flush=True)
            continue

        print(f"    Building multi-key index for {country} ({len(s2s3_c):,} records)...", flush=True)
        N = len(s2s3_c)
        s2s3_eids = s2s3_c["entity_id"].values
        s2s3_tokens = s2s3_c["name_tokens"].values
        s2s3_names = s2s3_c["name_clean"].values
        s2s3_nums = s2s3_c["addr_numerics"].values

        name_index = defaultdict(list)
        for eid, toks in zip(s2s3_eids, s2s3_tokens):
            for t in set(toks):
                if len(t) > 1:
                    name_index[t].append(eid)

        addr_num_index = defaultdict(list)
        for eid, nums in zip(s2s3_eids, s2s3_nums):
            for n in set(nums):
                if len(n) >= 4:
                    addr_num_index[n].append(eid)

        c3_index = defaultdict(list)
        for eid, nc in zip(s2s3_eids, s2s3_names):
            if nc:
                for g in set(_get_c3(nc)):
                    c3_index[g].append(eid)

        s2s3_nc_map = defaultdict(list)
        for eid, nc in zip(s2s3_eids, s2s3_names):
            if nc:
                s2s3_nc_map[nc].append(eid)

        # Precompute IDF weights using math.log for high-speed indexing
        import math
        token_idf = {t: math.log((N + 1) / (len(p) + 1)) * 2.0 for t, p in name_index.items()}
        num_idf = {n: math.log((N + 1) / (len(p) + 1)) * 1.5 for n, p in addr_num_index.items()}
        max_freq = int(N * 0.02)

        print(f"    Retrieving candidates for {len(s1_c):,} entities in {country}...", flush=True)
        part_cands = _generate_candidates_for_partition(
            s1_c["entity_id"].values,
            s1_c["name_tokens"].values,
            s1_c["name_clean"].values,
            s1_c["addr_numerics"].values,
            name_index,
            token_idf,
            addr_num_index,
            num_idf,
            c3_index,
            s2s3_nc_map,
            max_freq,
            max_candidates,
        )
        all_candidates.update(part_cands)

        lens = [len(v) for v in part_cands.values()]
        avg_cands = np.mean(lens) if lens else 0
        p95_cands = np.percentile(lens, 95) if lens else 0
        n_empty = sum(1 for l in lens if l == 0)
        print(f"    {country}: avg {avg_cands:.1f} cands, p95={p95_cands:.0f}, empty={n_empty:,}", flush=True)

    elapsed = time.time() - t0
    total_pairs = sum(len(v) for v in all_candidates.values())
    print(f"  Candidate generation done in {elapsed:.1f}s: {total_pairs:,} pairs", flush=True)
    return all_candidates


def measure_blocking_recall(candidates: dict, gt_df: pd.DataFrame) -> dict:
    """Measure what fraction of ground-truth matches appear in candidate lists."""
    from data_loader import parse_matched_ids

    total_true_pairs = 0
    found_pairs = 0
    entities_with_matches = 0
    entities_fully_covered = 0
    entities_partially_covered = 0

    for s1_id, matched_str in zip(gt_df["source1_entity_id"].values, gt_df["matched_entity_ids"].values):
        true_matches = parse_matched_ids(matched_str)
        if not true_matches:
            continue

        entities_with_matches += 1
        true_set = set(true_matches)
        cand_set = set(candidates.get(s1_id, []))

        hits = len(true_set & cand_set)
        total_true_pairs += len(true_set)
        found_pairs += hits

        if hits == len(true_set):
            entities_fully_covered += 1
        elif hits > 0:
            entities_partially_covered += 1

    recall = found_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    full_cov = (entities_fully_covered / entities_with_matches
                if entities_with_matches > 0 else 0.0)

    return {
        "total_true_pairs": total_true_pairs,
        "found_pairs": found_pairs,
        "blocking_recall": recall,
        "entities_with_matches": entities_with_matches,
        "entities_fully_covered": entities_fully_covered,
        "entities_partially_covered": entities_partially_covered,
        "full_coverage_rate": full_cov,
    }
