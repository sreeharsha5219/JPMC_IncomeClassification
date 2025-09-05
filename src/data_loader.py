from __future__ import annotations

"""
Data loading utilities specific to the CPS census dataset.

Thought process:
- Read everything as string initially so we don't accidentally coerce special
  tokens like '?' or 'Not in universe' into NaN/0 at this stage.
- Keep parsing rules transparent here; defer numeric coercion to the
  preprocessing step where we can handle errors consistently.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def read_columns(columns_path: str | Path) -> List[str]:
    """Read header names line-by-line.

    The provided header file lists one column per line aligned with the data.
    We keep ordering intact and ignore blank lines.
    """
    cols = []
    with open(columns_path, "r", encoding="utf-8") as fh:
        for line in fh:
            name = line.strip()
            if name:
                cols.append(name)
    return cols


def load_census_data(data_path: str | Path, columns_path: str | Path) -> pd.DataFrame:
    """
    Load the raw CSV using provided headers, preserving raw tokens.
    We intentionally read as strings and prevent pandas from converting '?' to
    NaN automatically, so that we can treat these tokens consistently in the
    preprocessing pipeline.

    """
    columns = read_columns(columns_path)
    df = pd.read_csv(
        data_path,
        header=None,
        names=columns,
        dtype=str,
        na_values=[],  # do not auto-convert '?' to NaN yet
        keep_default_na=False,
    )
    return df


def split_features_target_weight(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split raw dataframe into features X, binary target y, and sample weights w.

    Label parsing: official labels vary slightly (e.g., '- 50000.' vs '50000+.').
    The rule of thumb we use is: if the raw string contains a '+', we treat it
    as ≥$50k (1); otherwise it's <$50k (0). This is simple and robust to minor
    formatting differences observed in the file.
    """
    if "label" not in df.columns or "weight" not in df.columns:
        raise ValueError("Expected columns 'weight' and 'label' in data")

    X = df.drop(columns=["label", "weight"])  # features only
    y_raw = df["label"].astype(str).str.strip()
    y = y_raw.apply(lambda s: 1 if "+" in s or "50000+." in s or "50000+" in s else 0)

    # Sample weights represent population counts; keep as float for sklearn.
    w = pd.to_numeric(df["weight"], errors="coerce")
    return X, y, w


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Heuristic split into numeric vs categorical columns by name.

    a small set of fields are clearly numeric even though the file
    is read as strings. We explicitly list these by canonical header text to
    avoid guessing based on content (which could be fragile with tokens like
    'Not in universe').
    The rest are treated as categoricals by default.
    """
    numeric_like = {
        "age",
        "wage per hour",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "num persons worked for employer",
        "family members under 18",
        "weeks worked in year",
        "year",
    }
    numeric_cols = [c for c in df.columns if c in numeric_like]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols







