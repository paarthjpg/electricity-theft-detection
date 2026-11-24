# Electricity Theft Detection Using Graph Neural Networks — Project Report

## 1. Abstract

Electricity theft costs utilities billions of dollars annually and is
difficult to detect from individual consumption records alone. This project
builds a complete, reproducible pipeline that represents electricity
customers as nodes in a **Customer Consumption Similarity Graph** — built
from engineered features of their historical smart-meter readings, not
physical grid topology — and trains Graph Neural Networks (GCN, GraphSAGE)
to classify customers as normal or theft, under a leakage-safe inductive
evaluation protocol. Results are compared honestly against conventional ML
baselines (Logistic Regression, Random Forest, XGBoost) on the real SGCC
dataset (42,285 customers after cleaning, 1,034 days, 8.5% theft
prevalence). On this dataset and split, **XGBoost (PR-AUC 0.346) and Random
Forest (PR-AUC 0.311) outperformed all GNN variants (PR-AUC 0.256-0.283)**,
though GraphSAGE's real similarity graph did outperform a randomized-edge
control of the same size, suggesting the graph carries some, but limited,
useful signal beyond the engineered features themselves.

## 2. Introduction

Smart meters generate dense, longitudinal consumption records that make
data-driven theft detection feasible at scale. Most prior work treats each
customer independently; this project investigates whether relational
information — similarity between customers' consumption *profiles* — adds
predictive value when modeled explicitly with GNNs.

## 3. Problem Statement

Binary node classification over electricity customers: predict whether a
customer is a normal consumer (label 0) or an electricity theft case (label
1), using historical daily consumption and relationships to similar
customers, producing a probabilistic risk score for investigation
prioritization.

## 4. Motivation

Utilities have limited inspection capacity; a model that ranks customers by
theft likelihood lets inspectors prioritize the highest-risk cases. Ranking
metrics (Precision@K) are therefore as important as standard classification
metrics.

## 5. Objectives

1. Build a reproducible, leakage-safe data pipeline for the SGCC dataset.
2. Engineer discriminative customer-level consumption features.
3. Construct an efficient, documented Customer Consumption Similarity Graph.
4. Train and fairly compare GCN, GraphSAGE, and conventional ML baselines.
5. Evaluate honestly with imbalance-aware metrics and ablation studies.
6. Ship a working, artifact-driven analytics dashboard.

## 6. Literature Review

- Zheng, Z., Yang, Y., Niu, X., Dai, H.-N., & Zhou, Y. (2018). *Wide and
  Deep Convolutional Neural Networks for Electricity-Theft Detection to
  Secure Smart Grids.* IEEE Transactions on Industrial Informatics, 14(4),
  1606-1615. (Original SGCC dataset release and CNN-based baseline.)
- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with
  Graph Convolutional Networks.* ICLR. (GCN architecture used here.)
- Hamilton, W., Ying, R., & Leskovec, J. (2017). *Inductive Representation
  Learning on Large Graphs.* NeurIPS. (GraphSAGE architecture and the
  inductive aggregation principle this project's leakage-safe protocol
  relies on.)

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

## 10. Graph Construction Methodology

A k-nearest-neighbor (k=5 default, cosine distance) **Customer Consumption
Similarity Graph** over engineered features — explicitly not physical grid
topology (unverified for this dataset). Training graph (29,599 nodes):
212,902 directed / 106,451 undirected edges, average degree 14.39, 0
isolated nodes, 50 connected components (largest: 28,079 nodes), density
0.0002, mean similarity 0.979. See `METHODOLOGY.md` §1-2 for the full
leakage-safe inductive construction protocol.

## 11. GCN Architecture

Two `GCNConv` layers (64 -> 32 hidden units, self-loops added, symmetric
normalization), ReLU, dropout 0.3, linear classification head. Evaluated
under the documented inductive protocol (§ METHODOLOGY.md §2), not a
separate transductive shortcut.

## 12. GraphSAGE Architecture

Two `SAGEConv` layers (64 -> 32 hidden units, mean neighbor aggregation),
ReLU, dropout 0.3, linear classification head. The primary candidate for
inductive inference on genuinely unseen customers.

## 13. Baseline ML Algorithms

Logistic Regression (balanced class weights, C grid-searched on validation),
Random Forest (balanced class weights, n_estimators/max_depth grid-searched),
and XGBoost (`scale_pos_weight` from training labels, grid-searched) — all
trained on the identical standardized features and splits as the GNNs.

## 14. Experimental Setup

Adam optimizer, lr=0.001, weight_decay=0.0005, up to 100 epochs, early
stopping (patience 15) on validation PR-AUC, `BCEWithLogitsLoss` with
`pos_weight` from training labels only, seed 42 throughout, CPU execution
(Apple M4, MPS intentionally not used — see `src/training/train_gnn.py` for
why). QUICK MODE (stratified ~4,000-customer subset) available for fast
iteration; all results below are FULL MODE (complete 42,285-customer
dataset).

## 15. Evaluation Metrics

Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
confusion matrix, false positive/negative rate, Precision@K (K=5%, 10%).
Classification thresholds are chosen on validation data only (F1-optimized),
never on the test set.

## 16. Results and Discussion

See [`RESULTS.md`](RESULTS.md) (auto-generated by `scripts/run_evaluation.py`
directly from the saved evaluation JSON files — not hand-written) for the
full comparison table. Summary, ranked by test PR-AUC:

1. XGBoost — 0.3464 (F1 0.3691)
2. Random Forest — 0.3109 (F1 0.3613)
3. GraphSAGE k=3 — 0.2827 (F1 0.3392)
4. GraphSAGE k=5 — 0.2822 (F1 0.3443)
5. GraphSAGE k=10 — 0.2821 (F1 0.3411)
6. GraphSAGE k=5, randomized edges (control) — 0.2601 (F1 0.3026)
7. GCN k=5 — 0.2562 (F1 0.2953)
8. Logistic Regression — 0.2502 (F1 0.2917)

Tree-based baselines (XGBoost, Random Forest) outperformed both GNNs on
this dataset and split. GraphSAGE did outperform its randomized-edge control
(0.282 vs. 0.260 PR-AUC), indicating the specific similarity structure
carries some real signal beyond an arbitrary graph of the same size — but
not enough to close the gap with XGBoost. k had little effect on GraphSAGE
(k=3/5/10 all within 0.0006 PR-AUC of each other).

## 17. Ablation Studies

| Configuration | PR-AUC | F1 | Recall |
|---|---|---|---|
| A: Random Forest (no graph) | 0.3109 | 0.3613 | 0.4164 |
| B: GraphSAGE k=3 | 0.2827 | 0.3392 | 0.3420 |
| C: GraphSAGE k=5 | 0.2822 | 0.3443 | 0.3587 |
| D: GraphSAGE k=10 | 0.2821 | 0.3411 | 0.3401 |
| E: GCN (documented inductive protocol) | 0.2562 | 0.2953 | 0.2862 |
| F: GraphSAGE, randomized edges (control) | 0.2601 | 0.3026 | 0.3420 |

All ablations use the identical seed, split, and features; only the graph
configuration changes. Full configuration/protocol documentation is in
`METHODOLOGY.md`.

## 18. Limitations

- The similarity graph is feature-space similarity, not verified physical
  grid topology.
- ~25.6% of raw readings are missing and were imputed per-customer; this can
  understate volatility for heavily-missing customers.
- Only two-layer GCN/GraphSAGE architectures were evaluated (deeper
  architectures, attention-based variants, and learned graph structures
  were out of scope).
- Utility-reported theft labels may themselves contain noise from the
  original inspection process, which the model inherits.
- CPU-only training (MPS was found to have unsupported/unstable scatter
  kernels for the pinned PyG version); full-dataset training of each GNN
  configuration took ~2.5-5 minutes on an Apple M4 laptop, which is fine for
  this project's scale but would need GPU/CUDA for much larger graphs.

## 19. Future Scope

- Learned or attention-weighted graph structures (e.g. GAT) instead of a
  fixed kNN graph.
- Deeper or residual GNN architectures to test whether oversmoothing (not
  graph uselessness) explains the GNN/baseline gap.
- Temporal GNNs that model the consumption sequence directly rather than
  only summary features.
- Ensembling GNN embeddings with XGBoost on engineered features.

## 20. Conclusion

This project delivers a complete, honestly-evaluated pipeline answering its
research question: on the SGCC dataset, graph-based learning via GCN/
GraphSAGE did **not** outperform a well-tuned XGBoost baseline using the same
engineered features, though the specific similarity graph did carry some
measurable signal over a random graph of equivalent size. This is reported
as a genuine, non-adjusted finding consistent with known GNN limitations
(weak homophily, class imbalance, shallow architectures) rather than an
endorsement of GNNs as unconditionally superior for this task.

## 21. References

1. Zheng, Z., Yang, Y., Niu, X., Dai, H.-N., & Zhou, Y. (2018). Wide and Deep
   Convolutional Neural Networks for Electricity-Theft Detection to Secure
   Smart Grids. *IEEE Transactions on Industrial Informatics*, 14(4), 1606-1615.
2. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with
   Graph Convolutional Networks. *ICLR*.
3. Hamilton, W., Ying, R., & Leskovec, J. (2017). Inductive Representation
   Learning on Large Graphs. *NeurIPS*.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
   *Journal of Machine Learning Research*, 12, 2825-2830.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting
   System. *KDD*.
6. Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with
   PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs
   and Manifolds*.
