# Methodology

## 1. Feature Engineering

Each customer $i$ is represented by a feature vector $x_i \in \mathbb{R}^{21}$
— 21 engineered statistics of their daily consumption series, computed
vectorized across the full matrix (no per-customer loops): mean, median, std,
min, max, coefficient of variation, % zero days, % missing, trend slope,
monthly mean/variability, weekday/weekend ratio, sudden-drop frequency,
longest low-consumption run, first-vs-second-half ratio, rolling-mean
volatility, mean rolling std, seasonal variation, p10/p90 percentiles, and
long-term stability. See `src/data/feature_engineering.py`.

## 2. Train/Val/Test Split and Leakage Prevention

Customers are split 70/15/15 (train/val/test), stratified by label, with a
fixed seed (42), before any feature scaling or model fitting.

1. Train/val/test split happens on raw customer IDs before any preprocessing fitting.
2. The `StandardScaler` for engineered features is `.fit()` on training customers only (`scripts/prepare_data.py`).
3. Model hyperparameter selection (`src/training/train_baselines.py`) uses validation PR-AUC only; the test set is touched once per model, at the very end.
4. Classification thresholds (`src/evaluation/metrics.py::select_threshold_from_validation`) are chosen using validation scores/labels only, never the test set.
5. Per-customer missing-value imputation (median) uses only that customer's own series — no information crosses customers or splits.

## 3. Models

Three conventional ML models are trained on the same features and splits for
a fair comparison (`src/models/baselines.py`):

- **Logistic Regression** — linear baseline, `class_weight="balanced"`, grid search over regularization strength $C \in \{0.01, 0.1, 1.0, 10.0\}$.
- **Random Forest** — `class_weight="balanced"`, grid over `n_estimators` $\in \{100, 300\}$ and `max_depth` $\in \{\text{None}, 12\}$.
- **XGBoost** — gradient-boosted trees with `scale_pos_weight` set from the training class ratio, grid over `n_estimators` $\in \{100, 300\}$ and `max_depth` $\in \{4, 6\}$.

For each model, every grid candidate is fit on the training split and scored
on the validation split by PR-AUC; the best-scoring candidate is refit and
evaluated once on the test split.

## 4. Loss / Objective

Logistic Regression and Random Forest use scikit-learn's balanced
class-weighting (each sample reweighted inversely to its class frequency in
the training split) to counteract the ~8.5% theft prevalence. XGBoost uses
`scale_pos_weight = (negative count) / (positive count)` computed from the
training labels only.

## 5. Evaluation Metrics

Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC (average
precision), confusion matrix, false positive/negative rate, and
Precision@K / Recall@K at K=5%/10% (operationally relevant given limited
inspection capacity). See `src/evaluation/metrics.py`.

PR-AUC is the primary metric: with ~8.5% theft prevalence, a trivial
all-negative classifier already scores over 91% accuracy while catching zero
theft cases, so accuracy and even ROC-AUC can be misleadingly high. PR-AUC
and Precision@K are sensitive to how well the model ranks the rare positive
class, which is what matters when a utility can only physically inspect a
limited number of flagged customers.
