from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.ui_helpers import load_config, load_processed_artifacts, load_predictor, get_paths, missing_artifacts_notice
from src.graph.graph_builder import build_training_graph
from src.graph.graph_visualizer import sample_subgraph, plot_subgraph

config = load_config()
paths = get_paths(config)
data = load_processed_artifacts(paths["processed_dir"])

st.header("Graph Visualization")
st.info(
    "**Graph edges represent inferred consumption similarity, not verified physical "
    "electricity-grid connections.** Nodes are customers; edges connect the k most "
    "similar consumption profiles (k-nearest-neighbors in engineered feature space)."
)

if data is None:
    missing_artifacts_notice("Processed dataset")
    st.stop()

selected_tag = st.session_state.get("selected_model_tag")
predictor = None
if selected_tag:
    predictor = load_predictor(paths["processed_dir"], paths["models_dir"], paths["metrics_dir"], config, selected_tag)

customer_ids = data["customer_ids"]
train_rows = data["splits"]["train"]

color_mode = st.radio("Color nodes by", ["Ground-truth label", "Predicted risk"], horizontal=True)
max_nodes = st.slider("Max nodes in sampled subgraph", 20, 300, 120, step=10)

seed_customer = st.selectbox("Seed customer (center of the sampled neighborhood)", customer_ids[train_rows], index=0)
seed_row = predictor.id_to_row[seed_customer] if predictor else int(np.where(data["customer_ids"] == seed_customer)[0][0])

k = config["graph"]["k"]
train_features = data["features_scaled"][train_rows]

@st.cache_resource
def _cached_train_graph(processed_dir_str: str, k_: int):
    return build_training_graph(train_features, k=k_, metric=config["graph"]["metric"])

graph = _cached_train_graph(paths["processed_dir"], k)
seed_local = int(np.where(train_rows == seed_row)[0][0])

sub_g = sample_subgraph(graph, [seed_local], max_nodes=max_nodes)

node_labels_map = {local_idx: str(customer_ids[train_rows[local_idx]]) for local_idx in sub_g.nodes()}

if color_mode == "Ground-truth label":
    color_values = {local_idx: float(data["labels"][train_rows[local_idx]]) for local_idx in sub_g.nodes()}
    color_label = "Ground truth (1=Theft)"
elif predictor is not None:
    global_rows = np.array([train_rows[i] for i in sub_g.nodes()])
    scores = predictor.predict_rows_batch(global_rows)
    color_values = {local_idx: float(s) for local_idx, s in zip(sub_g.nodes(), scores)}
    color_label = "Predicted theft probability"
else:
    color_values = {n: 0.0 for n in sub_g.nodes()}
    color_label = "N/A (no model selected)"

st.plotly_chart(plot_subgraph(sub_g, color_values, node_labels_map, color_label), use_container_width=True)

st.subheader("Selected Customer's Nearest Neighbors")
neighbor_mask = graph.edge_index[0] == seed_local
neighbor_locals = graph.edge_index[1][neighbor_mask].tolist()
neighbor_weights = graph.edge_weight[neighbor_mask].tolist()
import pandas as pd
neighbor_df = pd.DataFrame({
    "Neighbor Customer ID": [customer_ids[train_rows[n]] for n in neighbor_locals],
    "Similarity": neighbor_weights,
    "Ground Truth": ["Theft" if data["labels"][train_rows[n]] == 1 else "Normal" for n in neighbor_locals],
}).sort_values("Similarity", ascending=False)
st.dataframe(neighbor_df, use_container_width=True, hide_index=True)

st.caption(f"Sampled subgraph: {sub_g.number_of_nodes()} nodes, {sub_g.number_of_edges()} edges "
           f"(full training graph has {graph.n_nodes:,} nodes — never rendered in full in the browser).")
