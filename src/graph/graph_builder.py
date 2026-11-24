"""Customer Consumption Similarity Graph construction.

IMPORTANT: the SGCC dataset carries no verified physical grid topology.
Edges here encode feature-space similarity between customers' engineered
consumption profiles, nothing else. Never describe them as physical
transformer/feeder/transmission-line connections.

Leakage-safe, inductive protocol:
  - `build_training_graph` builds a symmetric kNN graph using ONLY training
    customers and ONLY their features (no val/test features or labels).
  - `build_inductive_edges` connects each held-out (val/test) customer to its
    k nearest TRAINING customers only, as one-directional edges
    (train -> held-out). Held-out customers never gain edges to each other
    and never feed information back into the training graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@dataclass
class GraphBundle:
    edge_index: torch.Tensor  # shape (2, n_edges), long
    edge_weight: torch.Tensor  # shape (n_edges,), float — similarity in [0, 1]
    n_nodes: int


def _cosine_similarity_from_distance(distance: np.ndarray) -> np.ndarray:
    # sklearn's "cosine" metric returns 1 - cosine_similarity
    return np.clip(1.0 - distance, 0.0, 2.0)


def build_training_graph(train_features: np.ndarray, k: int, metric: str = "cosine") -> GraphBundle:
    """Symmetric kNN graph over training customers only.

    Node indices in the returned edge_index are positions into
    `train_features` (0..n_train-1) — callers are responsible for mapping
    these back to global customer indices if needed.
    """
    n_train = train_features.shape[0]
    k_query = min(k + 1, n_train)  # +1 because a point is its own 1st neighbor
    nn_model = NearestNeighbors(n_neighbors=k_query, metric=metric, algorithm="auto")
    nn_model.fit(train_features)
    distances, indices = nn_model.kneighbors(train_features)

    src_list, dst_list, w_list = [], [], []
    for i in range(n_train):
        for dist, j in zip(distances[i], indices[i]):
            if j == i:
                continue
            src_list.append(i)
            dst_list.append(j)
            w_list.append(_cosine_similarity_from_distance(np.array(dist)).item())

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    w = np.array(w_list, dtype=np.float32)

    # Symmetrize: a kNN relation is not inherently mutual (i's neighbor j
    # does not guarantee j's neighbor is i), so union both directions and
    # drop exact duplicate directed edges.
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    all_w = np.concatenate([w, w])
    pairs = np.stack([all_src, all_dst], axis=1)
    pairs_unique, unique_idx = np.unique(pairs, axis=0, return_index=True)
    w_unique = all_w[unique_idx]

    edge_index = torch.tensor(pairs_unique.T, dtype=torch.long)
    edge_weight = torch.tensor(w_unique, dtype=torch.float32)
    return GraphBundle(edge_index=edge_index, edge_weight=edge_weight, n_nodes=n_train)


def build_inductive_edges(
    query_features: np.ndarray,
    train_features: np.ndarray,
    k: int,
    metric: str = "cosine",
    train_offset: int = 0,
    query_offset: int | None = None,
) -> GraphBundle:
    """One-directional edges train -> query, for held-out (val/test) nodes.

    `train_offset` / `query_offset` let callers place train and query nodes
    into one shared global node-index space (query_offset defaults to
    train_offset + n_train, i.e. query nodes appended right after train
    nodes).
    """
    n_train = train_features.shape[0]
    if query_offset is None:
        query_offset = train_offset + n_train

    nn_model = NearestNeighbors(n_neighbors=min(k, n_train), metric=metric, algorithm="auto")
    nn_model.fit(train_features)
    distances, indices = nn_model.kneighbors(query_features)

    n_query = query_features.shape[0]
    src = (indices + train_offset).astype(np.int64).ravel()
    dst = (np.repeat(np.arange(n_query), indices.shape[1]) + query_offset).astype(np.int64)
    w = _cosine_similarity_from_distance(distances).astype(np.float32).ravel()

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)
    n_nodes = query_offset + n_query
    return GraphBundle(edge_index=edge_index, edge_weight=edge_weight, n_nodes=n_nodes)


def combine_graphs(train_graph: GraphBundle, inductive_graph: GraphBundle) -> GraphBundle:
    """Merge a train-train graph with train->held-out edges into one graph
    over the combined node space (assumes consistent offsets, see callers).
    """
    edge_index = torch.cat([train_graph.edge_index, inductive_graph.edge_index], dim=1)
    edge_weight = torch.cat([train_graph.edge_weight, inductive_graph.edge_weight], dim=0)
    n_nodes = max(train_graph.n_nodes, inductive_graph.n_nodes)
    return GraphBundle(edge_index=edge_index, edge_weight=edge_weight, n_nodes=n_nodes)


def randomize_edges(graph: GraphBundle, seed: int = 42) -> GraphBundle:
    """Graph-information control: shuffle which nodes edges connect while
    preserving the degree sequence and edge count. Used for the ablation
    that asks whether the *specific* similarity structure matters, not just
    having "some" graph. Never uses labels.
    """
    rng = np.random.default_rng(seed)
    n_edges = graph.edge_index.shape[1]
    nodes = np.arange(graph.n_nodes)
    new_src = rng.choice(nodes, size=n_edges, replace=True)
    new_dst = rng.choice(nodes, size=n_edges, replace=True)
    edge_index = torch.tensor(np.stack([new_src, new_dst], axis=0), dtype=torch.long)
    edge_weight = graph.edge_weight.clone()
    return GraphBundle(edge_index=edge_index, edge_weight=edge_weight, n_nodes=graph.n_nodes)
