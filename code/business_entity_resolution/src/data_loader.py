"""
data_loader.py — Reliable TSV loading with schema validation.

Loads Source 1/2/3 and ground-truth files, validates expected columns
and entity-ID uniqueness, and returns clean DataFrames.
"""

import os
import pandas as pd


# Expected columns for source files and ground truth
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def load_source_tsv(path: str, expected_prefix: str = None) -> pd.DataFrame:
    """Load a source TSV file (S1/S2/S3) and validate its schema.

    Args:
        path: Absolute or relative path to the TSV file.
        expected_prefix: If provided, assert every entity_id starts with this
                         prefix (e.g. 'S1-', 'S2-', 'S3-').

    Returns:
        A DataFrame with columns [entity_id, business_name, business_address, country].

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If required columns are missing or IDs are not unique.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str)

    # Validate columns
    missing_cols = set(SOURCE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{path}: missing required columns {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # Validate entity_id uniqueness
    n_ids = df["entity_id"].nunique()
    if n_ids != len(df):
        n_dups = len(df) - n_ids
        raise ValueError(
            f"{path}: entity_id is not unique — {n_dups:,} duplicate IDs found."
        )

    # Validate prefix if specified
    if expected_prefix is not None:
        bad = df[~df["entity_id"].str.startswith(expected_prefix)]
        if len(bad) > 0:
            samples = bad["entity_id"].head(5).tolist()
            raise ValueError(
                f"{path}: {len(bad):,} entity IDs do not start with "
                f"'{expected_prefix}'. Examples: {samples}"
            )

    # Fill NaN business_name/business_address with empty strings for safety
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")

    print(f"  Loaded {path}: {len(df):,} rows, "
          f"countries={df['country'].value_counts().to_dict()}")
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load the training ground-truth TSV.

    Returns a DataFrame with columns [source1_entity_id, matched_entity_ids].
    The matched_entity_ids column is kept as a raw string (comma-separated or
    empty/NaN for singletons).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str)

    missing_cols = set(GT_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{path}: missing required columns {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # Fill NaN matched_entity_ids with empty string (singletons)
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")

    print(f"  Loaded ground truth {path}: {len(df):,} S1 entities")
    return df


def parse_matched_ids(matched_str: str) -> list:
    """Parse the comma-separated matched_entity_ids string into a list.

    Args:
        matched_str: e.g. 'S2-00047,S2-00193,S3-00812' or '' for no matches.

    Returns:
        A list of matched entity IDs, or an empty list for singletons.
    """
    if not matched_str or matched_str.strip() == "":
        return []
    return [x.strip() for x in matched_str.split(",") if x.strip()]


def load_all_train(dataset_dir: str):
    """Load all training data files and return (s1, s2, s3, gt) DataFrames.

    Args:
        dataset_dir: Path to the dataset root (containing train/ and test/).

    Returns:
        Tuple of (train_s1, train_s2, train_s3, train_gt) DataFrames.
    """
    train_dir = os.path.join(dataset_dir, "train")
    print("Loading training data...")
    s1 = load_source_tsv(os.path.join(train_dir, "train_source1.tsv"), "S1-")
    s2 = load_source_tsv(os.path.join(train_dir, "train_source2.tsv"), "S2-")
    s3 = load_source_tsv(os.path.join(train_dir, "train_source3.tsv"), "S3-")
    gt = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))
    return s1, s2, s3, gt


def load_all_test(dataset_dir: str):
    """Load all test data files and return (s1, s2, s3) DataFrames.

    Args:
        dataset_dir: Path to the dataset root (containing train/ and test/).

    Returns:
        Tuple of (test_s1, test_s2, test_s3) DataFrames.
    """
    test_dir = os.path.join(dataset_dir, "test")
    print("Loading test data...")
    s1 = load_source_tsv(os.path.join(test_dir, "test_source1.tsv"), "S1-")
    s2 = load_source_tsv(os.path.join(test_dir, "test_source2.tsv"), "S2-")
    s3 = load_source_tsv(os.path.join(test_dir, "test_source3.tsv"), "S3-")
    return s1, s2, s3
