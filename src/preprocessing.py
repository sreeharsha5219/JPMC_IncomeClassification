from __future__ import annotations

"""
Preprocessing pipeline construction.

Thought process:
- Keep numeric handling explicit: coerce-to-numeric first so weird tokens like
  'Not in universe' become NaN, then median-impute and scale.
- For categoricals, convert '?' into NaN, impute with most frequent (simple,
  stable), then one-hot encode with handle_unknown=ignore so inference is safe.
- Support scikit-learn API differences (sparse vs sparse_output) to make the
  code portable across environments.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    # Numeric block: coerce -> impute -> scale
    num_pipe = Pipeline(
        steps=[
            ("coerce", _ToNumeric()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    # Categorical block: map '?' -> NaN -> impute -> one-hot
    cat_imputer = SimpleImputer(strategy="most_frequent")
    one_hot = _make_one_hot()

    cat_pipe = Pipeline(
        steps=[
            ("replace_unknown", _QuestionMarkToNa()),
            ("imputer", cat_imputer),
            ("onehot", one_hot),
        ]
    )

    # Combine into a single ColumnTransformer to keep feature assembly clean.
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )
    return pre


class _QuestionMarkToNa:
    """Small transformer to replace literal '?' tokens with NaN.

    We keep this as an explicit step so it's easy to reason about missingness
    and adjust later (e.g., mapping other sentinel values) without touching the
    core imputer configuration.
    """
    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.replace({"?": np.nan})
        else:
            X = np.where(X == "?", np.nan, X)
        return X


def _make_one_hot() -> OneHotEncoder:
    """Build a OneHotEncoder compatible with sklearn versions.

    sklearn >=1.2 renamed 'sparse' to 'sparse_output'. We try the new API
    first, and gracefully fall back to the older parameter if needed.
    """
    try:
        # scikit-learn >=1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # scikit-learn <1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class _ToNumeric:
    """
    Coerce dataframe/array to numeric dtype, non-parsable -> NaN.

    This keeps all numeric features on a clear path to imputation and scaling
    while being resilient to string tokens present in the raw file.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            for c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            return X
        # assume numpy array
        df = pd.DataFrame(X)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.values
