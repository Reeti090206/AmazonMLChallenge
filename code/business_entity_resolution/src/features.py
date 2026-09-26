"""
features.py — Pairwise feature extraction for candidate entity pairs.

Computes string-similarity features between an S1 record and a candidate
S2/S3 record.  All features are derived solely from the three shared
fields (business_name, business_address, country) and are country-agnostic.
"""

import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


# ---------------------------------------------------------------------------
# Low-level similarity helpers
# ---------------------------------------------------------------------------

def _jaccard_tokens(tokens_a: list, tokens_b: list) -> float:
    """Word-level Jaccard similarity between two token lists."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    sa, sb = set(tokens_a), set(tokens_b)
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _overlap_count(tokens_a: list, tokens_b: list) -> int:
    """Number of shared tokens between two lists."""
    if not tokens_a or not tokens_b:
        return 0
    return len(set(tokens_a) & set(tokens_b))


def _char_ngram_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity (spaces removed)."""
    def ngrams(s):
        s = s.replace(" ", "")
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    if not text_a and not text_b:
        return 1.0
    if not text_a or not text_b:
        return 0.0
    ga, gb = ngrams(text_a), ngrams(text_b)
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0


def _length_diff_ratio(text_a: str, text_b: str) -> float:
    """Normalised absolute length difference: |len(a)-len(b)| / max(len(a),len(b))."""
    la, lb = len(text_a), len(text_b)
    mx = max(la, lb)
    return abs(la - lb) / mx if mx else 0.0


def _numeric_overlap(nums_a: list, nums_b: list) -> int:
    """Count of shared purely-numeric tokens (postal codes, building nos.)."""
    if not nums_a or not nums_b:
        return 0
    return len(set(nums_a) & set(nums_b))


# ---------------------------------------------------------------------------
# Single-pair feature vector
# ---------------------------------------------------------------------------

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


def compute_pair_features(
    s1_name_clean: str, s1_name_tokens: list,
    s1_addr_clean: str, s1_addr_tokens: list, s1_addr_numerics: list,
    s1_country: str,
    s2_name_clean: str, s2_name_tokens: list,
    s2_addr_clean: str, s2_addr_tokens: list, s2_addr_numerics: list,
    s2_country: str,
) -> np.ndarray:
    """Compute the feature vector for a single (S1, S2/S3) candidate pair.

    Returns:
        1-D numpy array of length len(FEATURE_NAMES).
    """
    # Name features
    name_exact = 1.0 if (s1_name_clean and s1_name_clean == s2_name_clean) else 0.0
    name_jaccard = _jaccard_tokens(s1_name_tokens, s2_name_tokens)
    name_c3j = _char_ngram_jaccard(s1_name_clean, s2_name_clean, 3)
    name_fuzz = fuzz.ratio(s1_name_clean, s2_name_clean) / 100.0
    name_partial = fuzz.partial_ratio(s1_name_clean, s2_name_clean) / 100.0
    name_tok_ov = _overlap_count(s1_name_tokens, s2_name_tokens)
    name_len_d = _length_diff_ratio(s1_name_clean, s2_name_clean)

    # Address features
    addr_exact = 1.0 if (s1_addr_clean and s1_addr_clean == s2_addr_clean) else 0.0
    addr_jaccard = _jaccard_tokens(s1_addr_tokens, s2_addr_tokens)
    addr_c3j = _char_ngram_jaccard(s1_addr_clean, s2_addr_clean, 3)
    addr_fuzz = fuzz.ratio(s1_addr_clean, s2_addr_clean) / 100.0
    addr_tok_ov = _overlap_count(s1_addr_tokens, s2_addr_tokens)
    addr_num_ov = _numeric_overlap(s1_addr_numerics, s2_addr_numerics)
    addr_len_d = _length_diff_ratio(s1_addr_clean, s2_addr_clean)

    # Country feature
    country_m = 1.0 if s1_country == s2_country else 0.0

    return np.array([
        name_exact, name_jaccard, name_c3j, name_fuzz, name_partial,
        name_tok_ov, name_len_d,
        addr_exact, addr_jaccard, addr_c3j, addr_fuzz,
        addr_tok_ov, addr_num_ov, addr_len_d,
        country_m,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

def extract_features_batch(
    s1_records: dict,   # {entity_id: {name_clean, name_tokens, addr_clean, ...}}
    s2s3_records: dict, # same structure
    candidate_pairs: list,  # [(s1_id, s2s3_id), ...]
) -> np.ndarray:
    """Compute feature matrix for a batch of candidate pairs.

    Args:
        s1_records: dict mapping S1 entity_id to a dict of preprocessed fields.
        s2s3_records: dict mapping S2/S3 entity_id to a dict of preprocessed fields.
        candidate_pairs: list of (s1_id, s2s3_id) tuples.

    Returns:
        2-D numpy array of shape (len(candidate_pairs), len(FEATURE_NAMES)).
    """
    n = len(candidate_pairs)
    if n == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    features = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)

    for i, (s1_id, s2_id) in enumerate(candidate_pairs):
        r1 = s1_records[s1_id]
        r2 = s2s3_records[s2_id]
        features[i] = compute_pair_features(
            r1["name_clean"], r1["name_tokens"],
            r1["addr_clean"], r1["addr_tokens"], r1["addr_numerics"],
            r1["country_clean"],
            r2["name_clean"], r2["name_tokens"],
            r2["addr_clean"], r2["addr_tokens"], r2["addr_numerics"],
            r2["country_clean"],
        )

        if (i + 1) % 500_000 == 0:
            print(f"    Features computed: {i + 1:,}/{n:,}")

    return features


def df_to_record_dict(df) -> dict:
    """Convert a preprocessed DataFrame into a {entity_id: record_dict} lookup.

    The record dict contains: name_clean, name_tokens, addr_clean,
    addr_tokens, addr_numerics, country_clean.
    """
    keys = ["name_clean", "name_tokens", "addr_clean", "addr_tokens", "addr_numerics", "country_clean"]
    cols = [df[k].values for k in keys]
    eids = df["entity_id"].values
    return {eid: dict(zip(keys, vals)) for eid, *vals in zip(eids, *cols)}

