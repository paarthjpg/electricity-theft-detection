from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.ui_helpers import load_config, load_metrics, load_processed_artifacts, get_paths, missing_artifacts_notice
from src.graph.graph_builder import build_training_graph
from src.graph.graph_analysis import compute_graph_statistics

config = load_config()
paths = get_paths(config)
data = load_processed_artifacts(paths["processed_dir"])
metrics_bundle = load_metrics(paths["metrics_dir"])
table, results = metrics_bundle["table"], metrics_bundle["results"]

st.header("Research Insights")

if data is None:
    missing_artifacts_notice("Processed dataset")
    st.stop()

st.subheader("Feature Importance")
rf_report = results.get("random_forest")
if rf_report and rf_report.get("feature_importances"):
    imp_df = pd.DataFrame({"Feature": data["feature_names"], "Importance": rf_report["feature_importances"]}) \
        .sort_values("Importance", ascending=False)
    fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", template="plotly_white")
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Random Forest Gini importances — the only model here with a native, non-approximated "
               "feature-importance measure.")
else:
    missing_artifacts_notice("Random Forest results (for feature importance)")

st.subheader("Graph Statistics")
train_rows = data["splits"]["train"]

@st.cache_data
def _graph_stats(processed_dir_str: str, k: int):
    feats = data["features_scaled"][train_rows]
    graph = build_training_graph(feats, k=k, metric=config["graph"]["metric"])
    return compute_graph_statistics(graph)

stats = _graph_stats(paths["processed_dir"], config["graph"]["k"])
stat_cols = st.columns(4)
stat_items = list(stats.items())
for i, (key, value) in enumerate(stat_items):
    stat_cols[i % 4].metric(key.replace("_", " ").title(), f"{value:.4f}" if isinstance(value, float) else f"{value:,}")

st.subheader("Model Comparison Conclusions")
if table:
    ranked = sorted(((n, r.get("pr_auc") or 0) for n, r in table.items()), key=lambda p: p[1], reverse=True)
    best_name, best_score = ranked[0]
    st.markdown(f"Best model by test PR-AUC: **{best_name}** ({best_score:.4f}).")
    st.dataframe(pd.DataFrame(ranked, columns=["Model", "PR-AUC"]), use_container_width=True, hide_index=True)
else:
    missing_artifacts_notice("Model comparison results")

st.subheader("Graph Ablation Results")
ablation_tags = ["graphsage_k3", "graphsage_k5", "graphsage_k10", "graphsage_k5_randomized", "gcn_k5", "random_forest"]
ablation_rows = []
for tag in ablation_tags:
    if tag in results:
        r = results[tag]
        ablation_rows.append({"Configuration": tag, "PR-AUC": r.get("pr_auc"), "F1": r.get("f1"),
                               "Recall": r.get("recall")})
if ablation_rows:
    st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True, hide_index=True)
    st.caption(
        "A: Random Forest (no graph) · B/C/D: GraphSAGE at k=3/5/10 · E: GCN (documented inductive protocol) · "
        "F: GraphSAGE with randomized edges (same degree sequence, no label information — a graph-information "
        "control). See RESULTS.md for the full written comparison."
    )
else:
    missing_artifacts_notice("Ablation study results")

st.subheader("Examples of High-Risk Consumption Patterns")
labels = data["labels"]
theft_rows = np.where(labels == 1)[0]
if len(theft_rows):
    sample = np.random.default_rng(0).choice(theft_rows, size=min(3, len(theft_rows)), replace=False)
    dates = pd.to_datetime(data["dates"])
    for row in sample:
        series = data["reading_matrix_raw"][row]
        fig = px.line(x=dates, y=series, template="plotly_white",
                      labels={"x": "Date", "y": "Reading"},
                      title=f"Customer {data['customer_ids'][row]} (ground-truth: Theft)")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Known Limitations")
st.markdown(
    """
- The similarity graph encodes **feature-space similarity**, not verified physical grid topology — it may not
  reflect real electrical relationships between customers.
- Labels come from utility-reported theft flags, which themselves may contain false positives/negatives from
  the original inspection process; the model inherits any such label noise.
- ~25% of raw daily readings are missing and were imputed with per-customer medians, which can understate
  genuine volatility for heavily-missing customers.
- Two-layer GCN/GraphSAGE architectures were evaluated; deeper architectures or learned (rather than kNN)
  graph structures were out of scope for this project.
- Predictions are probabilistic risk indicators for prioritizing investigation — never proof of theft.
"""
)

st.subheader("Research Methodology Summary")
st.markdown(
    """
Customers are nodes in a kNN similarity graph built from 21 engineered consumption features.
Training/validation/test splits are customer-level, stratified, and seeded; the graph used during training
contains only training customers, and held-out customers are connected only to training-reference customers
during evaluation (inductive protocol) — see `METHODOLOGY.md` for the full write-up, including the GCN/GraphSAGE
architectures, loss function, and leakage-prevention measures.
"""
)
