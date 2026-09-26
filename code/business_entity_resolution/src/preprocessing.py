"""
preprocessing.py — Text normalisation for business names and addresses.

Produces normalised columns alongside the originals. All transformations
are country-agnostic (open-set safe) and use only information derivable
from the supplied data.
"""

import re
import unicodedata
import numpy as np
import pandas as pd

_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_RE_SPACES = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Lowercase, remove non-alphanumeric (keep spaces), collapse whitespace."""
    if not text:
        return ""
    text = text.lower()
    if not text.isascii():
        normed = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in normed if not unicodedata.combining(c))
    text = _RE_NON_ALNUM.sub(" ", text)
    return _RE_SPACES.sub(" ", text).strip()


def _tokenise(text: str) -> list:
    """Split normalised text into word tokens."""
    if not text:
        return []
    return text.split()


def _extract_numeric_tokens(text: str) -> list:
    """Extract all purely numeric tokens from a string (postal codes, building nos)."""
    if not text:
        return []
    return [t for t in text.split() if t.isdigit()]


def preprocess_dataframe(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """Add normalised columns to a source DataFrame.

    New columns created:
        name_clean      — lowercased, unicode-normalised, punctuation-stripped name
        name_tokens     — list of word tokens from name_clean
        addr_clean      — lowercased, unicode-normalised, punctuation-stripped address
        addr_tokens     — list of word tokens from addr_clean
        addr_numerics   — list of purely numeric tokens from addr_clean
        country_clean   — lowercased, stripped country string
    """
    if not inplace:
        df = df.copy()

    # Clean names
    df["name_clean"] = df["business_name"].fillna("").apply(_clean_text)
    df["name_tokens"] = df["name_clean"].apply(_tokenise)

    # Clean addresses
    df["addr_clean"] = df["business_address"].fillna("").apply(_clean_text)
    df["addr_tokens"] = df["addr_clean"].apply(_tokenise)
    df["addr_numerics"] = df["addr_clean"].apply(_extract_numeric_tokens)

    # Country — just lowercase and strip
    df["country_clean"] = df["country"].fillna("").str.lower().str.strip()

    return df
