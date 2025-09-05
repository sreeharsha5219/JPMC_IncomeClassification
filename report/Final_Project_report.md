# Income Classification & Customer Segmentation — Final Report

Author: Sree Harsha Koyi  
Date: 2025‑09‑04  
Audience: Retail marketing and CRM stakeholders; data science reviewers

---

## Executive Summary

- Goal: predict income group (< $50k vs ≥ $50k) and segment the population into actionable personas.
- Data: CPS 1994–1995 weighted survey; 40 features + year + label; 199,523 rows.
- Best classifier: XGBoost (hist) with a preprocessing pipeline (one‑hot + scaled numerics).
- Test metrics (weighted): ROC AUC ≈ 0.956, Accuracy ≈ 0.956, Precision ≈ 0.743, Recall ≈ 0.492, F1 ≈ 0.592.
- Threshold trade‑off: lowering threshold from 0.50 to ≈ 0.26 improves recall to ≈ 0.69 and F1 to ≈ 0.63 with modest accuracy impact.
- Segmentation: KMeans (k=6) yields interpretable personas; silhouette ≈ 0.19.
- Deliverables: code in `src/`, metrics/plots in `reports/`, model in `models/`, segments and profiles under `outputs/` and `reports/`.

---

## 1) Data Understanding

- Source & shape: 199,523 rows × 42 columns (40 features + `year` + `label`).
- Types: mixed numeric (e.g., age, wages, capital gains/losses, weeks worked) and categorical (education, industry, occupation, marital status, etc.).
- Weights: each row has a `weight` (population representativeness). Used for training and evaluation (`sample_weight`), never as a feature.
- Label mapping: strings containing `+` → 1 (≥ $50k), others → 0 (< $50k). Class is imbalanced toward < $50k (weighted).
- Missingness: categoricals use `?`; numeric‑like fields can contain tokens like “Not in universe”.

Artifacts (weighted EDA):
- Missingness by column: `reports/eda/missingness.csv`
- Numeric summaries: `reports/eda/numeric_summary.csv`
- Categorical distributions (top categories): `reports/eda/categorical_distributions.csv`

---

## 2) Preprocessing & Feature Engineering

- Numeric: coerce to numeric → median imputation → standard scaling.
- Categorical: map `?` → NA → most‑frequent imputation → one‑hot with `handle_unknown=ignore` (safe inference on new data).
- Feature assembly: implemented via a `ColumnTransformer` + `Pipeline` to keep preprocessing inside the model artifact.
- Optional dimensionality controls:
  - Mutual information feature selection (top 50%).
  - TruncatedSVD (100 components) to reduce variance from high‑dimensional one‑hot features.

Code references: `src/preprocessing.py`, `src/data_loader.py`.

---

## 3) Modeling Approach

- Candidates (all wrapped in the same preprocessing; optional FS/FE as above):
  - Logistic Regression (L2)
  - Random Forest (`n_estimators=300`)
  - XGBoost (`n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `tree_method=hist`)
- Selection: 5‑fold stratified CV on the training split using weighted ROC AUC; pick best CV, refit on full training, evaluate on hold‑out test.
- Rationale: ROC AUC is threshold‑independent and weight‑aware; balanced comparison under class imbalance.

Code reference: `src/train_classifier.py`.

---

## 4) Classification Results

- Best by CV: XGBoost without extra reduction (`xgb__none`).
- Test metrics (weighted) — from `reports/classification_metrics.json`:
  - ROC AUC ≈ 0.956
  - Accuracy ≈ 0.956
  - Precision ≈ 0.743
  - Recall ≈ 0.492
  - F1 ≈ 0.592
- Threshold sweep (see `reports/threshold_metrics.csv` and `reports/plots/threshold_metrics.png`):
  - Around threshold ≈ 0.26: Recall ≈ 0.69, F1 ≈ 0.63 (useful for broad campaigns).
  - Use higher thresholds (≈ 0.50–0.65) for precision‑driven, high‑cost channels.
- Directional feature importance (global permutation; `reports/permutation_importance.csv`):
  - Age, weeks worked, tax filer status, education, capital gains/dividends rank among the most influential — face‑valid socio‑economic drivers.

Key plots/artifacts:
- ROC curve: `reports/plots/roc_curve.png`
- Confusion matrix (weighted counts): `reports/plots/confusion_matrix.png`
- Metrics JSON: `reports/classification_metrics.json`

---

## 5) Segmentation

- Method: KMeans on the preprocessed feature space (one‑hot + scaled numerics).
- Chosen k: 6 (balance interpretability vs differentiation); silhouette ≈ 0.19 (reasonable for high‑dimensional, mixed‑type data).
- Persona highlights:
  - High‑Earning Professionals — higher education, capital income → premium/financial offers.
  - Prime Working‑Age Retail/Clerical (two clusters) — value‑oriented offers, loyalty programs.
  - Full‑Time Hourly (long weeks) — practical goods, tools, durable products.
  - Older Adults Not in Labor Force — healthcare, retirement, convenience/leisure.
  - Dependents/Children — non‑earners; only relevant for household‑level marketing.

Artifacts:
- Assignments: `outputs/segments.csv`
- Weighted segment profiles: `reports/segment_profiles.csv`

---

## 6) Recommendations

- Targeting & thresholds: rank by predicted probability; pick thresholds by cost and channel.
  - Broad reach: threshold ≈ 0.25–0.35 to lift recall.
  - High‑CPM precision: threshold ≈ 0.50–0.65 to favor precision.
- Segment‑tailored strategy: overlay classifier ranks with personas to tailor creative, offers, and channels.
- Measurement: A/B tests with control groups; monitor conversion, AOV, and LTV uplift.
- Fairness & compliance: monitor subgroup metrics and calibration across sensitive proxies; adjust thresholds/post‑processing per policy.
- Lifecycle: monitor drift; retrain on fresher data; recalibrate thresholds periodically.

---

## 7) Risks & Limitations

- Data recency: CPS 1994–1995; validate on contemporary data before production.
- $50k threshold: historical; consider inflation‑adjusted or continuous income targets.
- Fairness: demographic correlates require careful governance in marketing applications.

---

## 8) Implementation Notes

- Code: `src/train_classifier.py`, `src/segment.py`, `src/preprocessing.py`, `src/data_loader.py`, `src/eda.py`
- Artifacts: `reports/` (metrics, plots, threshold table, importance), `models/classifier.joblib` (full pipeline), `outputs/segments.csv`, `reports/segment_profiles.csv`
- Reproducibility: set `--seed`; pinned dependencies in `requirements.txt`.

---

## 9) References

- scikit‑learn User Guide (pipelines, preprocessing, model evaluation)
- Hastie, Tibshirani, Friedman — The Elements of Statistical Learning
- Kaufman & Rousseeuw — Finding Groups in Data (silhouette)

---

## 10) Conclusion

The classifier provides strong ranking performance with clear threshold trade‑offs, and the segmentation offers coherent personas for tailored messaging. Combined, they deliver a practical foundation for targeting, budgeting, and creative strategy — ready to pilot with current data and live A/B testing.

