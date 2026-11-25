# Electricity Theft Detection Using Machine Learning — Project Report

## 1. Abstract

Electricity theft costs utilities billions of dollars annually and is
difficult to detect from individual consumption records alone. This project
builds a complete, reproducible pipeline that engineers 21 statistical
features from each customer's historical smart-meter readings and trains
three conventional ML models — Logistic Regression, Random Forest, and
XGBoost — to classify customers as normal or theft, under a leakage-safe
evaluation protocol. Evaluated on the real SGCC dataset (42,285 customers
after cleaning, 1,034 days, 8.5% theft prevalence), **XGBoost achieves the
best test PR-AUC (0.3464)**, ahead of Random Forest (0.3109) and Logistic
Regression (0.2502).

## 2. Introduction

Smart meters generate dense, longitudinal consumption records that make
data-driven theft detection feasible at scale. This project investigates
which conventional ML model best separates theft from normal usage using
engineered summary statistics of each customer's consumption history, under
the severe class imbalance typical of real-world theft-detection data.

## 3. Problem Statement

Binary classification over electricity customers: predict whether a customer
is a normal consumer (label 0) or an electricity theft case (label 1), using
historical daily consumption, producing a probabilistic risk score for
investigation prioritization.

## 4. Motivation

Utilities have limited inspection capacity; a model that ranks customers by
theft likelihood lets inspectors prioritize the highest-risk cases. Ranking
metrics (Precision@K) are therefore as important as standard classification
metrics.

## 5. Objectives

1. Build a reproducible, leakage-safe data pipeline for the SGCC dataset.
2. Engineer discriminative customer-level consumption features.
3. Train and fairly compare Logistic Regression, Random Forest, and XGBoost.
4. Evaluate honestly with imbalance-aware metrics.
5. Ship a working, artifact-driven analytics dashboard.

## 6. Literature Review

- Zheng, Z., Yang, Y., Niu, X., Dai, H.-N., & Zhou, Y. (2018). *Wide and
  Deep Convolutional Neural Networks for Electricity-Theft Detection to
  Secure Smart Grids.* IEEE Transactions on Industrial Informatics, 14(4),
  1606-1615. (Original SGCC dataset release and CNN-based baseline.)
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting
  System.* KDD. (Primary model architecture used here.)
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.

## 7. Dataset Description

SGCC Electricity Theft Detection Dataset (see
[`data/raw/DATA_SOURCE.md`](data/raw/DATA_SOURCE.md) for the exact source
URL, citation, and SHA-256 checksum of the file used). Raw: 42,372
customers, daily readings 2014-01-01 to 2016-10-31 (1,034 days), 38,757
normal / 3,615 theft (10.72:1 imbalance). After cleaning (dropping 87
customers with <10% reading coverage): **42,285 customers**.

## 8. Data Preprocessing

- Negative readings (0 found) and z-score>6 extreme readings (4,040 of
  ~43.8M readings, 0.009%) are treated as missing, not capped, to avoid
  injecting a biased value.
- 25.64% of raw daily readings are missing; genuine zero consumption (13.3%
  of readings) is preserved as a real value, distinct from missing.
- Per-customer median imputation (never across customers).
- Customer-level stratified 70/15/15 train/val/test split, seed 42: 29,599 /
  6,343 / 6,343 customers, theft rate preserved at ~8.5% in every split.

## 9. Feature Engineering

21 engineered features per customer (see `src/data/feature_engineering.py`):
mean/median/std/min/max consumption, coefficient of variation, % zero days,
% missing, linear trend slope, average monthly consumption, monthly
variability, weekday/weekend ratio, sudden-drop frequency, longest
low-consumption run, first-half-vs-second-half ratio, rolling-mean
volatility, mean rolling std, seasonal variation, p10/p90 percentiles, and
long-term stability. All computed vectorized across the full matrix (no
per-customer Python loops). Features are standardized with a `StandardScaler`
fit on training customers only.

## 10. Models

Logistic Regression (balanced class weights, C grid-searched on validation),
Random Forest (balanced class weights, n_estimators/max_depth grid-searched),
and XGBoost (`scale_pos_weight` from training labels, grid-searched) — all
trained on identical standardized features and splits. See `METHODOLOGY.md`
§3 for the exact grids.

## 11. Experimental Setup

Each model's hyperparameter grid is fit on the training split and scored on
the validation split by PR-AUC; the best candidate is refit and evaluated
once on the test split. Seed 42 throughout. QUICK MODE (stratified
~4,000-customer subset) available for fast iteration; all results below are
FULL MODE (complete 42,285-customer dataset).

## 12. Evaluation Metrics

Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
confusion matrix, false positive/negative rate, Precision@K (K=5%, 10%).
Classification thresholds are chosen on validation data only (F1-optimized),
never on the test set.

## 13. Results and Discussion

See [`RESULTS.md`](RESULTS.md) (auto-generated by `scripts/run_evaluation.py`
directly from the saved evaluation JSON files — not hand-written) for the
full comparison table. Summary, ranked by test PR-AUC:

1. XGBoost — 0.3464 (F1 0.3691, accuracy 0.8868)
2. Random Forest — 0.3109 (F1 0.3613, accuracy 0.8751)
3. Logistic Regression — 0.2502 (F1 0.2917, accuracy 0.8660)

XGBoost's gradient-boosted trees best exploit the nonlinear interactions
among the 21 engineered features, followed by Random Forest's bagged-tree
ensemble; the linear Logistic Regression baseline trails both. Accuracy
alone is not meaningful here: with ~8.5% theft prevalence, a trivial
all-negative classifier already scores above 91% accuracy while catching no
theft at all, which is why PR-AUC is reported as the primary metric.

## 14. Limitations

- ~25.6% of raw readings are missing and were imputed per-customer; this can
  understate volatility for heavily-missing customers.
- Utility-reported theft labels may themselves contain noise from the
  original inspection process, which the model inherits.
- Only summary-statistic features of the consumption series are used; the
  raw daily sequence itself is not modeled directly.

## 15. Future Scope

- Sequence models (e.g. 1D-CNN, LSTM, or Transformer) over the raw daily
  consumption series directly, rather than only summary features.
- Richer feature engineering (e.g. frequency-domain features, customer
  metadata if available).
- Cost-sensitive threshold tuning aligned with actual inspection capacity
  and cost-per-inspection.

## 16. Conclusion

This project delivers a complete, honestly-evaluated pipeline answering its
research question: on the SGCC dataset, **XGBoost** best separates theft
from normal consumption among the three conventional ML models evaluated,
using only engineered summary-statistic features of each customer's
consumption history. Accuracy is a poor metric under this class imbalance,
so PR-AUC and Precision@K are emphasized throughout.

## 17. References

1. Zheng, Z., Yang, Y., Niu, X., Dai, H.-N., & Zhou, Y. (2018). Wide and Deep
   Convolutional Neural Networks for Electricity-Theft Detection to Secure
   Smart Grids. *IEEE Transactions on Industrial Informatics*, 14(4), 1606-1615.
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
   *Journal of Machine Learning Research*, 12, 2825-2830.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting
   System. *KDD*.
4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
