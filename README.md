# ML Project: Income Classification and Segmentation

This project trains a classifier to predict whether an individual's income is below $50,000 or at/above $50,000 using weighted U.S. Census data, and builds a segmentation (clustering) model for marketing personas.

## Data

- Input files in this folder:
  - `census-bureau.data` has comma-separated values, 40 features + `weight` + `label`.
  - `census-bureau.columns` - header names aligned with the data file. Note: the last two columns are `year` and `label`.

Missing values are represented with `?` in the data file.

## Environment

- Python 3.9+
- Recommended: create a virtual environment and install dependencies:

```
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Project Layout

```
src/
  __init__.py
  data_loader.py
  preprocessing.py
  train_classifier.py
  segment.py
reports/
  (generated: metrics, plots, segment profiles)
models/
  (generated: serialized classifier)
outputs/
  (generated: cluster assignments)
report/
  Final_Project_report.md
  JPMC_Risk_Assesment.pdf
```

## Deliverables

The key deliverables and their locations in this repo:

- Code: `src/train_classifier.py`, `src/segment.py`, `src/preprocessing.py`, `src/data_loader.py`, `src/eda.py`
- Report: `report/Project_Report.pdf` (PDF)
- Metrics: `reports/classification_metrics.json`
- Plots: `reports/plots/roc_curve.png`, `reports/plots/confusion_matrix.png`, `reports/plots/threshold_metrics.png`
- Threshold analysis: `reports/threshold_metrics.csv`
- Feature importance: `reports/permutation_importance.csv`
- Segments: `outputs/segments.csv`, `reports/segment_profiles.csv`
- Model artifact: `models/classifier.joblib`

## Usage

All commands assume the working directory is the project root (this folder).

1) Train and evaluate classifier (uses sample weights):

```
python -m src.train_classifier \
  --data_path census-bureau.data \
  --columns_path census-bureau.columns \
  --model_out models/classifier.joblib
```

Outputs:
- `reports/classification_metrics.json`
- `reports/plots/roc_curve.png`, `reports/plots/confusion_matrix.png`
- `reports/threshold_metrics.csv`, `reports/plots/threshold_metrics.png`
- `reports/permutation_importance.csv`
- `models/classifier.joblib`

Notes on modeling:
- The training script evaluates fixed pipelines (no hyperparameter tuning) that include feature selection (SelectPercentile with mutual information), feature extraction (TruncatedSVD), and classifiers (Logistic Regression, Random Forest, and XGBoost). The best model by 5-fold CV ROC AUC is chosen and saved.
- `reports/classification_metrics.json` also includes a short `candidates` list with CV scores for transparency.

2) Build segmentation model and profiles:

```
python -m src.segment \
  --data_path census-bureau.data \
  --columns_path census-bureau.columns \
  --k 6 \
  --assignments_out outputs/segments.csv
```

Outputs:
- `outputs/segments.csv` (record-level cluster IDs)
- `reports/segment_profiles.csv` (cluster summaries)

3) Run lightweight EDA (weighted distributions + missingness):

```
python -m src.eda \
  --data_path census-bureau.data \
  --columns_path census-bureau.columns \
  --out_dir reports/eda
```

Outputs:
- `reports/eda/missingness.csv`
- `reports/eda/numeric_summary.csv`
- `reports/eda/categorical_distributions.csv`

## Notes

- The classifier uses a preprocessing pipeline (numeric imputation + scaling, categorical one-hot) with Logistic Regression and Random Forest evaluated via weighted metrics; the best is reported.
- The segmentation uses KMeans on the same preprocessed feature space. You may adjust `--k` and review silhouette scores printed to console.
- The `weight` column is used as `sample_weight` for model training and for computing evaluation metrics. It is excluded from features.

## Reproducibility

- Set a random seed with `--seed` on both scripts to make splits and models deterministic.

## Reporting

- See `report/Final_Project_report.pdf` for the project report. it documents EDA, preprocessing, modeling choices, evaluation, and business recommendations.

