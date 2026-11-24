"""Graph diagnostics for a constructed similarity graph."""
from __future__ import annotations

import networkx as nx
import numpy as np
import torch

from src.graph.graph_builder import GraphBundle


def compute_graph_statistics(graph: GraphBundle) -> dict:
    edge_index = graph.edge_index.numpy()
    n_nodes = graph.n_nodes
    n_edges = edge_index.shape[1]

    degree = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(degree, edge_index[0], 1)
    np.add.at(degree, edge_index[1], 1)

    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    g.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    n_possible_edges = n_nodes * (n_nodes - 1) / 2
    density = g.number_of_edges() / n_possible_edges if n_possible_edges > 0 else 0.0

    return {
        "n_nodes": n_nodes,
        "n_edges_directed": int(n_edges),
        "n_edges_undirected": int(g.number_of_edges()),
        "avg_degree": float(degree.mean()),
        "median_degree": float(np.median(degree)),
        "max_degree": int(degree.max()) if n_nodes else 0,
        "n_isolated_nodes": int((degree == 0).sum()),
        "n_connected_components": nx.number_connected_components(g),
        "largest_component_size": max((len(c) for c in nx.connected_components(g)), default=0),
        "density": float(density),
        "similarity_mean": float(graph.edge_weight.mean().item()) if graph.edge_weight.numel() else 0.0,
        "similarity_std": float(graph.edge_weight.std().item()) if graph.edge_weight.numel() else 0.0,
    }


def validate_graph_integrity(graph: GraphBundle) -> dict:
    """Structural sanity checks used by tests: valid indices, no OOB, etc."""
    edge_index = graph.edge_index
    checks = {
        "indices_in_bounds": bool((edge_index >= 0).all() and (edge_index < graph.n_nodes).all()),
        "no_self_loops": bool((edge_index[0] != edge_index[1]).all()),
        "edge_weight_matches_edge_count": edge_index.shape[1] == graph.edge_weight.shape[0],
    }
    return checks
