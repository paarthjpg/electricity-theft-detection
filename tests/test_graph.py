import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.graph_builder import build_training_graph, build_inductive_edges, combine_graphs, randomize_edges
from src.graph.graph_analysis import compute_graph_statistics, validate_graph_integrity


def _toy_features(n, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


def test_training_graph_edges_are_in_bounds_and_symmetric():
    feats = _toy_features(40)
    graph = build_training_graph(feats, k=5)
    checks = validate_graph_integrity(graph)
    assert checks["indices_in_bounds"]
    assert checks["no_self_loops"]
    assert checks["edge_weight_matches_edge_count"]

    edge_set = set(map(tuple, graph.edge_index.t().tolist()))
    for (u, v) in list(edge_set)[:50]:
        assert (v, u) in edge_set, "training graph must be symmetric"


def test_no_duplicate_directed_edges():
    feats = _toy_features(40)
    graph = build_training_graph(feats, k=5)
    pairs = list(map(tuple, graph.edge_index.t().tolist()))
    assert len(pairs) == len(set(pairs))


def test_inductive_edges_connect_only_to_training_nodes():
    train_feats = _toy_features(50, seed=1)
    query_feats = _toy_features(10, seed=2)
    graph = build_inductive_edges(query_feats, train_feats, k=5, train_offset=0)
    src = graph.edge_index[0].numpy()
    dst = graph.edge_index[1].numpy()
    assert (src < 50).all(), "sources must all be training nodes"
    assert (dst >= 50).all(), "targets must all be held-out (query) nodes"
    assert graph.n_nodes == 60


def test_combined_graph_has_no_test_to_test_or_test_to_train_edges():
    train_feats = _toy_features(50, seed=1)
    val_feats = _toy_features(10, seed=3)
    train_graph = build_training_graph(train_feats, k=5)
    val_graph = build_inductive_edges(val_feats, train_feats, k=5, train_offset=0)
    combined = combine_graphs(train_graph, val_graph)
    dst_from_train_graph = set(train_graph.edge_index[1].tolist())
    assert all(d < 50 for d in dst_from_train_graph)
    # No edges originate FROM a val node (index >= 50) in the combined graph
    src_all = combined.edge_index[0].numpy()
    assert (src_all < 50).all()


def test_graph_statistics_reports_expected_fields():
    feats = _toy_features(60)
    graph = build_training_graph(feats, k=5)
    stats = compute_graph_statistics(graph)
    for key in ["n_nodes", "n_edges_undirected", "avg_degree", "n_isolated_nodes",
                "n_connected_components", "density"]:
        assert key in stats
    assert stats["n_nodes"] == 60


def test_randomized_edge_control_preserves_edge_count_not_structure():
    feats = _toy_features(60)
    graph = build_training_graph(feats, k=5)
    randomized = randomize_edges(graph, seed=42)
    assert randomized.edge_index.shape[1] == graph.edge_index.shape[1]
    assert randomized.n_nodes == graph.n_nodes
    original_pairs = set(map(tuple, graph.edge_index.t().tolist()))
    randomized_pairs = set(map(tuple, randomized.edge_index.t().tolist()))
    assert original_pairs != randomized_pairs
