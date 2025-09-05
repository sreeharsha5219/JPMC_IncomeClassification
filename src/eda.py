from __future__ import annotations

"""
Lightweight EDA for weighted distributions and missingness.

Outputs under reports/eda/:
- missingness.csv — un/weighted missingness per column
- numeric_summary.csv — weighted mean/std and quantiles for numeric-like columns
- categorical_distributions.csv — top categories by weighted count per categorical column
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .data_loader import infer_feature_types, load_census_data, split_features_target_weight


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    m = np.average(x.fillna(0.0), weights=w.reindex_like(x).fillna(0.0))
    return float(m)


def weighted_std(x: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    w = w.reindex_like(x).fillna(0.0)
    avg = np.average(x.fillna(0.0), weights=w)
    var = np.average((x.fillna(0.0) - avg) ** 2, weights=w)
    return float(np.sqrt(var))


def weighted_quantile(x: pd.Series, w: pd.Series, q: float) -> float:
    x = pd.to_numeric(x, errors="coerce")
    mask = x.notna() & w.notna()
    xv = x[mask].values
    wv = w[mask].values
    if len(xv) == 0:
        return float("nan")
    order = np.argsort(xv)
    xv_sorted = xv[order]
    w_sorted = wv[order]
    cum_w = np.cumsum(w_sorted)
    total = cum_w[-1]
    target = q * total
    idx = np.searchsorted(cum_w, target, side="left")
    idx = np.clip(idx, 0, len(xv_sorted) - 1)
    return float(xv_sorted[idx])


def compute_missingness(df: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    rows = []
    total = len(df)
    total_w = float(w.sum()) if float(w.sum()) > 0 else np.nan
    for col in df.columns:
        # Treat '?' and empty string as missing for EDA
        miss_mask = df[col].isin(["?", ""]) | df[col].isna()
        miss_count = int(miss_mask.sum())
        miss_w = float(w[miss_mask].sum())
        rows.append(
            {
                "column": col,
                "missing_count": miss_count,
                "missing_pct": miss_count / total if total > 0 else np.nan,
                "missing_weight": miss_w,
                "missing_weight_pct": miss_w / total_w if total_w and total_w > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("missing_weight", ascending=False)


def summarize_numeric(df: pd.DataFrame, cols: List[str], w: pd.Series) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = df[col]
        rows.append(
            {
                "column": col,
                "w_mean": weighted_mean(s, w),
                "w_std": weighted_std(s, w),
                "w_p10": weighted_quantile(s, w, 0.10),
                "w_p50": weighted_quantile(s, w, 0.50),
                "w_p90": weighted_quantile(s, w, 0.90),
            }
        )
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def summarize_categorical(df: pd.DataFrame, cols: List[str], w: pd.Series, top_k: int = 10) -> pd.DataFrame:
    out_rows = []
    total_w = float(w.sum()) if float(w.sum()) > 0 else 0.0
    for col in cols:
        grp = df.groupby(col)[w.name].sum().sort_values(ascending=False)
        grp = grp.head(top_k)
        for cat, wt in grp.items():
            out_rows.append(
                {
                    "column": col,
                    "category": str(cat),
                    "weighted_count": float(wt),
                    "weighted_share": float(wt / total_w) if total_w > 0 else np.nan,
                }
            )
    return pd.DataFrame(out_rows).sort_values(["column", "weighted_count"], ascending=[True, False])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="census-bureau.data")
    p.add_argument("--columns_path", default="census-bureau.columns")
    p.add_argument("--out_dir", default="reports/eda")
    args = p.parse_args()

    data = load_census_data(args.data_path, args.columns_path)
    X, y, w = split_features_target_weight(data)
    w.name = "__w"
    num_cols, cat_cols = infer_feature_types(X)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    miss = compute_missingness(X, w)
    miss.to_csv(out_dir / "missingness.csv", index=False)

    num_summary = summarize_numeric(X, num_cols, w)
    num_summary.to_csv(out_dir / "numeric_summary.csv", index=False)

    cat_df = X[cat_cols].copy()
    cat_df[w.name] = w.values
    cat_summary = summarize_categorical(cat_df, cat_cols, cat_df[w.name])
    cat_summary.to_csv(out_dir / "categorical_distributions.csv", index=False)

    print(f"Saved EDA outputs to {out_dir}")


if __name__ == "__main__":
    main()

