from __future__ import annotations

"""
Unsupervised segmentation using KMeans on the preprocessed feature space.

Thought process:
- KMeans is a sensible first pass after one-hot + scaling; it's fast and easy
  to interpret. We report a silhouette score to sanity-check cohesion.
- We keep post-hoc profiling simple and actionable: weighted means for a few
  numeric anchors and top categories for key demographic fields.
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from .data_loader import infer_feature_types, load_census_data, split_features_target_weight
from .preprocessing import build_preprocessor


# Chosen categorical fields to summarize per cluster. We keep it small to
# highlight the most interpretable dimensions for marketing.
KEY_CATEGORICAL = [
    "education",
    "marital stat",
    "major occupation code",
    "major industry code",
    "race",
    "sex",
    "citizenship",
]


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    """Compute a simple weighted mean, tolerant to non-numeric data."""
    x = pd.to_numeric(x, errors="coerce")
    m = np.average(x.fillna(0.0), weights=w.reindex_like(x).fillna(0.0))
    return float(m)


def build_profiles(df: pd.DataFrame, clusters: pd.Series, weights: pd.Series, numeric_cols: List[str]) -> pd.DataFrame:
    """Summarize each cluster: size, weighted share, numeric means, top categories.

    We intentionally keep aggregation lightweight and readable. For production,
    we might add more comprehensive distributions, but this already surfaces
    clear directions for marketing.
    """
    out_rows: List[Dict] = []
    df = df.copy()
    df["cluster"] = clusters.values
    df["__w"] = weights.values

    total_w = df["__w"].sum()
    for cl, grp in df.groupby("cluster"):
        row: Dict = {
            "cluster": int(cl),
            "count": int(len(grp)),
            "weighted_count": float(grp["__w"].sum()),
        }
        row["weighted_share"] = row["weighted_count"] / total_w if total_w > 0 else np.nan
        for col in numeric_cols:
            row[f"{col}__wmean"] = weighted_mean(grp[col], grp["__w"]) if col in grp.columns else np.nan

        # Selected categorical distributions (top category by weighted share)
        for col in KEY_CATEGORICAL:
            if col in grp.columns:
                wsum = grp.groupby(col)["__w"].sum().sort_values(ascending=False)
                if len(wsum) > 0:
                    top_cat = wsum.index[0]
                    top_share = float(wsum.iloc[0] / wsum.sum()) if wsum.sum() > 0 else np.nan
                    row[f"top_{col}"] = str(top_cat)
                    row[f"top_{col}_share"] = top_share
        out_rows.append(row)

    prof = pd.DataFrame(out_rows).sort_values("cluster").reset_index(drop=True)
    return prof


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="census-bureau.data")
    p.add_argument("--columns_path", default="census-bureau.columns")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--assignments_out", default="outputs/segments.csv")
    args = p.parse_args()

    data = load_census_data(args.data_path, args.columns_path)
    X_all, y_all, w_all = split_features_target_weight(data)
    num_cols, cat_cols = infer_feature_types(X_all)
    pre = build_preprocessor(num_cols, cat_cols)

    # Reuse the same preprocessing as the classifier (no leakage risk here).
    pipe = Pipeline(steps=[("pre", pre)])
    Z = pipe.fit_transform(X_all)  # numpy array

    kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10)
    kmeans.fit(Z)
    cl = kmeans.labels_

    sil = silhouette_score(Z, cl, metric="euclidean")
    print(f"KMeans k={args.k} silhouette_score={sil:.3f}")

    # Save assignments for record-level linkage back to the original data
    assign_df = pd.DataFrame({"cluster": cl})
    out_path = Path(args.assignments_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assign_df.to_csv(out_path, index=False)
    print(f"Saved cluster assignments to {out_path}")

    profiles = build_profiles(X_all, pd.Series(cl), w_all, num_cols)
    rep_dir = Path("reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(rep_dir / "segment_profiles.csv", index=False)
    print(f"Saved segment profiles to {rep_dir / 'segment_profiles.csv'}")


if __name__ == "__main__":
    main()
