from __future__ import annotations

"""
Train a weighted classifier with a reproducible pipeline and report metrics.

Thought process:
- Use a Pipeline so preprocessing is part of the model artifact (safer serving).
- Evaluate simple, strong baselines first (LogReg, RandomForest) with weighted
  ROC AUC — consistent with imbalanced data and threshold-free comparison.
- Hold out a test set, and also do 5-fold CV within training for model choice.
- Save plots + metrics for communication, and the fitted pipeline for reuse.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from .data_loader import infer_feature_types, load_census_data, split_features_target_weight
from .preprocessing import build_preprocessor

try:
    # Optional dependency; ensured via requirements.txt
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None  # allow import even if xgboost not installed


def evaluate(model: Pipeline, X, y, sample_weight=None) -> Dict:
    """Compute a small set of threshold-dependent metrics at p>=0.5.

    Note: For business deployment, we often tune the probability threshold.
    Here we report at 0.5 for comparability; downstream users can adjust.
    """
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y, y_proba, sample_weight=sample_weight)),
        "accuracy": float(accuracy_score(y, y_pred, sample_weight=sample_weight)),
        "precision": float(precision_score(y, y_pred, zero_division=0, sample_weight=sample_weight)),
        "recall": float(recall_score(y, y_pred, zero_division=0, sample_weight=sample_weight)),
        "f1": float(f1_score(y, y_pred, zero_division=0, sample_weight=sample_weight)),
    }
    return metrics


def plot_roc(model: Pipeline, X, y, out_path: Path, sample_weight=None):
    """Plot ROC curve and save with AUC in legend."""
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba, sample_weight=sample_weight)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(model: Pipeline, X, y, out_path: Path, sample_weight=None):
    """Plot weighted confusion matrix (absolute weighted counts)."""
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred, sample_weight=sample_weight)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format=".0f")
    plt.title("Confusion Matrix (weighted counts)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_threshold_metrics(
    model: Pipeline,
    X,
    y,
    out_csv: Path,
    out_plot: Path,
    sample_weight=None,
    steps: int = 101,
):
    """Compute precision/recall/accuracy/F1 across thresholds and save CSV + plot."""
    y_proba = model.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.0, 1.0, steps)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "precision": float(precision_score(y, y_pred, zero_division=0, sample_weight=sample_weight)),
                "recall": float(recall_score(y, y_pred, zero_division=0, sample_weight=sample_weight)),
                "f1": float(f1_score(y, y_pred, zero_division=0, sample_weight=sample_weight)),
                "accuracy": float(accuracy_score(y, y_pred, sample_weight=sample_weight)),
            }
        )
    # Save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd  # local import to avoid overhead at module import

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.plot(df["threshold"], df["accuracy"], label="Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold vs Metrics (test set)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()


def permutation_importance_report(
    model: Pipeline,
    X,
    y,
    out_csv: Path,
    seed: int,
):
    """Compute permutation importance on the full pipeline.

    Note: Uses unweighted ROC AUC for simplicity; interpreting magnitudes should
    be done cautiously. This provides directional importance for business use.
    """
    r = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=5,
        random_state=seed,
        n_jobs=-1,
    )
    import pandas as pd

    cols = list(X.columns)
    df = pd.DataFrame({
        "feature": cols,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

def _feature_stages(seed: int) -> List[Tuple[str, object]]:
    """Return minimal feature selection/extraction stages (no tuning).

    We evaluate three fixed choices to demonstrate tradeoffs without performing
    hyperparameter sweeps: passthrough, MI-based selection (50%), and SVD(100).
    """
    return [
        ("none", "passthrough"),
        ("select_pct50_mi", SelectPercentile(score_func=mutual_info_classif, percentile=50)),
        ("svd_100", TruncatedSVD(n_components=100, random_state=seed)),
    ]


def build_models(pre: ColumnTransformer, seed: int) -> List[Tuple[str, Pipeline]]:
    """Assemble candidate pipelines with fixed, sensible defaults (no tuning).

    Models: Logistic Regression (L2), RandomForest, and XGBoost (if available).
    Each is paired with the three fixed feature stages from `_feature_stages`.
    """
    models: List[Tuple[str, Pipeline]] = []
    for fs_name, fs in _feature_stages(seed):
        # Logistic Regression (L2, default strength)
        lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
        models.append((f"lr__{fs_name}", Pipeline(steps=[("pre", pre), ("fs", fs), ("clf", lr)])))

        # Random Forest (balanced capacity, stable default)
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed)
        models.append((f"rf__{fs_name}", Pipeline(steps=[("pre", pre), ("fs", fs), ("clf", rf)])))

        # XGBoost (single, well-behaved config)
        if XGBClassifier is not None:
            xgb = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="auc",
                random_state=seed,
                tree_method="hist",
                n_jobs=-1,
            )
            models.append((f"xgb__{fs_name}", Pipeline(steps=[("pre", pre), ("fs", fs), ("clf", xgb)])))
    return models


def cross_val_roc_auc(model: Pipeline, X, y, sample_weight, seed: int) -> float:
    """Simple 5-fold stratified CV with sample weights passed to the estimator."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores: List[float] = []
    for train_idx, valid_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[valid_idx]
        w_tr, w_va = None, None
        if sample_weight is not None:
            w_tr = sample_weight.iloc[train_idx]
            w_va = sample_weight.iloc[valid_idx]
        # Pass sample weights to the final estimator (named 'clf' in the Pipeline)
        model.fit(X_tr, y_tr, clf__sample_weight=w_tr)
        y_proba = model.predict_proba(X_va)[:, 1]
        score = roc_auc_score(y_va, y_proba, sample_weight=w_va)
        scores.append(float(score))
    return float(np.mean(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="census-bureau.data")
    p.add_argument("--columns_path", default="census-bureau.columns")
    p.add_argument("--model_out", default="models/classifier.joblib")
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    # Load and split raw dataset
    data = load_census_data(args.data_path, args.columns_path)
    X_all, y_all, w_all = split_features_target_weight(data)
    num_cols, cat_cols = infer_feature_types(X_all)
    pre = build_preprocessor(num_cols, cat_cols)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_all, y_all, w_all, test_size=0.2, random_state=args.seed, stratify=y_all
    )

    models = build_models(pre, seed=args.seed)

    best_name = None
    best_model = None
    best_cv = -np.inf
    candidates: List[Dict] = []
    for name, pipe in models:
        # Choose by CV ROC AUC on training only to avoid test leakage.
        cv_score = cross_val_roc_auc(pipe, X_train, y_train, w_train, seed=args.seed)
        candidates.append({"name": name, "cv_roc_auc": float(cv_score)})
        if cv_score > best_cv:
            best_cv = cv_score
            best_name = name
            best_model = pipe

    assert best_model is not None
    best_model.fit(X_train, y_train, clf__sample_weight=w_train)

    train_metrics = evaluate(best_model, X_train, y_train, sample_weight=w_train)
    test_metrics = evaluate(best_model, X_test, y_test, sample_weight=w_test)

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_roc(best_model, X_test, y_test, plots_dir / "roc_curve.png", sample_weight=w_test)
    plot_confusion(best_model, X_test, y_test, plots_dir / "confusion_matrix.png", sample_weight=w_test)

    # Threshold-metrics curve for business trade-offs
    plot_threshold_metrics(
        best_model,
        X_test,
        y_test,
        out_csv=out_dir / "threshold_metrics.csv",
        out_plot=plots_dir / "threshold_metrics.png",
        sample_weight=w_test,
    )

    # Permutation importance (global) for interpretability
    permutation_importance_report(
        best_model,
        X_test,
        y_test,
        out_csv=out_dir / "permutation_importance.csv",
        seed=args.seed,
    )

    metrics_payload = {
        "best_model": best_name,
        "cv_roc_auc": best_cv,
        "train": train_metrics,
        "test": test_metrics,
        "candidates": sorted(candidates, key=lambda d: d["cv_roc_auc"], reverse=True)[:20],
    }
    with open(out_dir / "classification_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, model_out)

    # Lightweight console summary for quick inspection
    print(json.dumps(metrics_payload, indent=2))
    print(f"Saved model to {model_out}")


if __name__ == "__main__":
    main()
