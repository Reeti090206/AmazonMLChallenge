"""
features.py — Pairwise feature extraction for candidate entity pairs.

Computes 15 string-similarity features between an S1 record and a candidate
S2/S3 record. Uses lightweight record tuples for minimal memory footprint (<100 MB).
"""

import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
    "name_exact",
    "name_jaccard",
    "name_char3_jaccard",
    "name_fuzz_ratio",
    "name_fuzz_partial",
    "name_token_overlap",
    "name_len_diff",
    "addr_exact",
    "addr_jaccard",
    "addr_char3_jaccard",
    "addr_fuzz_ratio",
    "addr_token_overlap",
    "addr_numeric_overlap",
    "addr_len_diff",
    "country_match",
]


def compute_pair_features(r1: tuple, r2: tuple) -> list:
    """Compute the 15-feature vector for a single pair using lightweight record tuples."""
    nc1, nt1, ac1, at1, an1, cc1 = r1
    nc2, nt2, ac2, at2, an2, cc2 = r2

    # Name features
    name_exact = 1.0 if (nc1 and nc1 == nc2) else 0.0

    s1_nt, s2_nt = set(nt1), set(nt2)
    inter_nt = len(s1_nt & s2_nt)
    union_nt = len(s1_nt | s2_nt)
    name_jaccard = inter_nt / union_nt if union_nt else (1.0 if not s1_nt and not s2_nt else 0.0)

    # Name Char 3-gram Jaccard
    nc1_s = nc1.replace(" ", "")
    nc2_s = nc2.replace(" ", "")
    s1_nc3 = {nc1_s[i:i + 3] for i in range(len(nc1_s) - 2)} if len(nc1_s) >= 3 else ({nc1_s} if nc1_s else set())
    s2_nc3 = {nc2_s[i:i + 3] for i in range(len(nc2_s) - 2)} if len(nc2_s) >= 3 else ({nc2_s} if nc2_s else set())
    inter_c3 = len(s1_nc3 & s2_nc3)
    union_c3 = len(s1_nc3 | s2_nc3)
    name_c3j = inter_c3 / union_c3 if union_c3 else (1.0 if not s1_nc3 and not s2_nc3 else 0.0)

    name_fuzz = fuzz.ratio(nc1, nc2) / 100.0
    name_partial = fuzz.partial_ratio(nc1, nc2) / 100.0
    name_tok_ov = float(inter_nt)

    mx_nl = max(len(nc1), len(nc2))
    name_len_d = abs(len(nc1) - len(nc2)) / mx_nl if mx_nl else 0.0

    # Address features
    addr_exact = 1.0 if (ac1 and ac1 == ac2) else 0.0

    s1_at, s2_at = set(at1), set(at2)
    inter_at = len(s1_at & s2_at)
    union_at = len(s1_at | s2_at)
    addr_jaccard = inter_at / union_at if union_at else (1.0 if not s1_at and not s2_at else 0.0)

    ac1_s = ac1.replace(" ", "")
    ac2_s = ac2.replace(" ", "")
    s1_ac3 = {ac1_s[i:i + 3] for i in range(len(ac1_s) - 2)} if len(ac1_s) >= 3 else ({ac1_s} if ac1_s else set())
    s2_ac3 = {ac2_s[i:i + 3] for i in range(len(ac2_s) - 2)} if len(ac2_s) >= 3 else ({ac2_s} if ac2_s else set())
    inter_ac3 = len(s1_ac3 & s2_ac3)
    union_ac3 = len(s1_ac3 | s2_ac3)
    addr_c3j = inter_ac3 / union_ac3 if union_ac3 else (1.0 if not s1_ac3 and not s2_ac3 else 0.0)

    addr_fuzz = fuzz.ratio(ac1, ac2) / 100.0
    addr_tok_ov = float(inter_at)

    addr_num_ov = float(len(set(an1) & set(an2)))

    mx_al = max(len(ac1), len(ac2))
    addr_len_d = abs(len(ac1) - len(ac2)) / mx_al if mx_al else 0.0

    country_m = 1.0 if cc1 == cc2 else 0.0

    return [
        name_exact, name_jaccard, name_c3j, name_fuzz, name_partial,
        name_tok_ov, name_len_d,
        addr_exact, addr_jaccard, addr_c3j, addr_fuzz,
        addr_tok_ov, addr_num_ov, addr_len_d,
        country_m,
    ]


def extract_features_batch(
    s1_records: dict,
    s2s3_records: dict,
    candidate_pairs: list,
) -> np.ndarray:
    """Compute feature matrix for a batch of candidate pairs."""
    n = len(candidate_pairs)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    features = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)

    for i, (s1_id, s2_id) in enumerate(candidate_pairs):
        r1 = s1_records[s1_id]
        r2 = s2s3_records[s2_id]
        features[i] = compute_pair_features(r1, r2)

    return features


def df_to_record_dict(df) -> dict:
    """Convert a preprocessed DataFrame into a lightweight {entity_id: 6-tuple} lookup.

    Tuple contents: (name_clean, name_tokens, addr_clean, addr_tokens, addr_numerics, country_clean)
    """
    records = {}
    eids = df["entity_id"].values
    nc_vals = df["name_clean"].values
    nt_vals = df["name_tokens"].values
    ac_vals = df["addr_clean"].values
    at_vals = df["addr_tokens"].values
    an_vals = df["addr_numerics"].values
    cc_vals = df["country_clean"].values

    for eid, nc, nt, ac, at, an, cc in zip(eids, nc_vals, nt_vals, ac_vals, at_vals, an_vals, cc_vals):
        records[eid] = (
            nc if isinstance(nc, str) else "",
            nt if isinstance(nt, list) else [],
            ac if isinstance(ac, str) else "",
            at if isinstance(at, list) else [],
            an if isinstance(an, list) else [],
            cc if isinstance(cc, str) else "",
        )

    return records
