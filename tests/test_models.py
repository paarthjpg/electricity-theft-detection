import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graph.graph_builder import build_training_graph, build_inductive_edges, combine_graphs
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE


def _toy_graph(n=40, d=10, seed=0):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, d)).astype(np.float32)
    graph = build_training_graph(feats, k=5)
    return feats, graph


def test_gcn_forward_pass_output_shape():
    feats, graph = _toy_graph()
    model = GCN(in_dim=feats.shape[1], hidden_dim_1=16, hidden_dim_2=8, dropout=0.3)
    x = torch.tensor(feats)
    logits = model(x, graph.edge_index, graph.edge_weight)
    assert logits.shape == (feats.shape[0],)


def test_graphsage_forward_pass_output_shape():
    feats, graph = _toy_graph()
    model = GraphSAGE(in_dim=feats.shape[1], hidden_dim_1=16, hidden_dim_2=8, dropout=0.3)
    x = torch.tensor(feats)
    logits = model(x, graph.edge_index)
    assert logits.shape == (feats.shape[0],)


def test_loss_is_finite_and_backprop_works():
    feats, graph = _toy_graph()
    y = torch.tensor(np.random.default_rng(1).integers(0, 2, size=feats.shape[0]), dtype=torch.float32)
    model = GraphSAGE(in_dim=feats.shape[1], hidden_dim_1=16, hidden_dim_2=8, dropout=0.3)
    x = torch.tensor(feats)
    logits = model(x, graph.edge_index)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert any(g > 0 for g in grad_norms)


def test_checkpoint_save_and_reload_gives_same_output():
    feats, graph = _toy_graph()
    model = GCN(in_dim=feats.shape[1], hidden_dim_1=16, hidden_dim_2=8, dropout=0.0)
    model.eval()
    x = torch.tensor(feats)
    with torch.no_grad():
        out_before = model(x, graph.edge_index, graph.edge_weight)

    state = model.state_dict()
    model2 = GCN(in_dim=feats.shape[1], hidden_dim_1=16, hidden_dim_2=8, dropout=0.0)
    model2.load_state_dict(state)
    model2.eval()
    with torch.no_grad():
        out_after = model2(x, graph.edge_index, graph.edge_weight)

    assert torch.allclose(out_before, out_after)


def test_inductive_inference_on_unseen_customers():
    """A trained model must be able to score customers it never saw during
    training, connected only to training-reference neighbors."""
    train_feats, train_graph = _toy_graph(n=50, seed=2)
    unseen_feats = np.random.default_rng(3).normal(size=(5, train_feats.shape[1])).astype(np.float32)

    model = GraphSAGE(in_dim=train_feats.shape[1], hidden_dim_1=16, hidden_dim_2=8, dropout=0.0)
    model.eval()

    unseen_graph = build_inductive_edges(unseen_feats, train_feats, k=5)
    combined = combine_graphs(train_graph, unseen_graph)
    x_all = torch.tensor(np.concatenate([train_feats, unseen_feats], axis=0))
    with torch.no_grad():
        logits = model(x_all, combined.edge_index)
    unseen_logits = logits[len(train_feats):]
    assert unseen_logits.shape == (5,)
    assert torch.isfinite(unseen_logits).all()
