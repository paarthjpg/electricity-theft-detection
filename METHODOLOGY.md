# Methodology

## 1. Customer Consumption Similarity Graph

Let there be $N$ customers. Each customer $i$ is a graph node with a feature
vector $x_i \in \mathbb{R}^{21}$ — 21 engineered statistics of their daily
consumption series (mean, median, std, min, max, coefficient of variation,
% zero days, % missing, trend slope, monthly mean/variability, weekday/weekend
ratio, sudden-drop frequency, longest low-consumption run, first-vs-second-half
ratio, rolling-mean volatility, mean rolling std, seasonal variation, p10/p90
percentiles, long-term stability).

Edges are defined by **k-nearest-neighbors in feature space**, not physical
grid topology (unverified for this dataset):

$$E = \{(i, j) : x_j \in \text{kNN}_k(x_i)\} \cup \{(j, i) : x_j \in \text{kNN}_k(x_i)\}$$

using cosine distance, $k=5$ by default (ablated at $k \in \{3, 5, 10\}$).
The union with its reverse symmetrizes the (not inherently mutual) kNN
relation. This is explicitly a **Customer Consumption Similarity Graph** —
rendering or dashboard code must never describe it as a verified
transformer/feeder/transmission-line connection.

## 2. Leakage-Safe Inductive Protocol

Customers are split 70/15/15 (train/val/test), stratified by label, with a
fixed seed (42), before any graph construction. Define:

- **Training graph** $G_{train} = (V_{train}, E_{train})$: kNN edges
  computed using only $\{x_i : i \in V_{train}\}$.
- **Inductive evaluation edges** for a held-out customer $h \in
  V_{val} \cup V_{test}$: $k$ directed edges $(t, h)$ for each of $h$'s $k$
  nearest neighbors $t \in V_{train}$ — never $(h, t)$, and never
  $(h, h')$ for another held-out $h'$.

A forward pass over $G_{train} \cup \{(t,h) : \ldots\}$ computes correct
layer-1 embeddings for training nodes (using only $E_{train}$, unaffected by
appended held-out nodes) and correct layer-2 embeddings for held-out nodes
(aggregating training nodes' updated layer-1 embeddings plus their own raw
features) — a standard, valid inductive inference procedure, not a
transductive shortcut. See `src/graph/graph_builder.py` and
`src/training/train_gnn.py::_eval_inductive`.

GCN is evaluated under this same documented inductive procedure (not a
separate transductive variant) because `GCNConv`'s normalized-adjacency
message passing is well-defined on any graph passed at call time, including
one with appended held-out nodes.

## 3. GCN Message Passing

$$H^{(l+1)} = \sigma\left(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

where $\hat{A} = A + I$ (self-loops added), $\hat{D}$ is the degree matrix of
$\hat{A}$. Two layers (64 -> 32 hidden units), ReLU, dropout 0.3, then a
linear classification head producing one logit per node.

## 4. GraphSAGE Neighborhood Aggregation

$$h_i^{(l+1)} = \sigma\left(W^{(l)} \cdot \text{CONCAT}\left(h_i^{(l)}, \text{MEAN}_{j \in N(i)} h_j^{(l)}\right)\right)$$

Mean aggregation over available neighbors generalizes naturally to nodes
unseen during training, since a held-out node's embedding only requires its
own features and its training-neighbors' features/embeddings — no retraining
or access to other held-out nodes needed. This makes GraphSAGE the primary
candidate for inductive inference in this project.

## 5. Loss Function and Class Imbalance

Binary cross-entropy with a positive-class weight computed from **training
labels only**:

$$\mathcal{L} = -\frac{1}{|V_{train}|}\sum_{i \in V_{train}} w_{pos} \cdot y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i), \quad w_{pos} = \frac{|\{i : y_i=0\}|}{|\{i : y_i=1\}|}$$

implemented as `torch.nn.BCEWithLogitsLoss(pos_weight=w_pos)`.

## 6. Inductive vs. Transductive Evaluation

**Inductive** (used for the headline comparison): held-out customers are
never part of the graph seen during training; at evaluation time they are
appended as new nodes connected only to training references (§2). This is
the honest test of generalization to genuinely new customers.

**Transductive** would mean the held-out nodes' features (not labels) are
already part of the static graph used throughout training — not used here
for the headline comparison, to avoid implying GCN/GraphSAGE had different
information access than they actually did.

## 7. Data Leakage Prevention Checklist

1. Train/val/test split happens on raw customer IDs before any preprocessing fitting.
2. The `StandardScaler` for engineered features is `.fit()` on training customers only (`scripts/prepare_data.py`).
3. The training graph (§2) is built from training customers' features only.
4. Held-out customers connect only to training references, never to each other.
5. Classification thresholds (`src/evaluation/metrics.py::select_threshold_from_validation`) are chosen using validation scores/labels only.
6. The test set is touched exactly once per model, for final reporting.
7. Per-customer missing-value imputation (median) uses only that customer's own series — no information crosses customers or splits.

## 8. Evaluation Metrics

Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC
(average precision), confusion matrix, false positive/negative rate,
Precision@K / Recall@K at K=5%/10% (operationally relevant given limited
inspection capacity). See `src/evaluation/metrics.py`. PR-AUC is emphasized
over accuracy/ROC-AUC because of the ~8.5% theft prevalence — see the
Model Performance dashboard page for the full explanation.
