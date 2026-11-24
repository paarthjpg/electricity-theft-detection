"""GCN for the Customer Consumption Similarity Graph.

GCNConv's normalized-adjacency message passing is inherently defined on a
fixed graph (it re-normalizes degree using the edges present in the forward
call). For a technically honest inductive evaluation, held-out customers are
appended as extra nodes with edges only FROM training nodes (see
src/graph/graph_builder.build_inductive_edges) and a forward pass is run
over that combined graph — training-node embeddings are computed using only
train-train edges (unaffected by the appended held-out nodes), so this is a
valid, documented inductive inference procedure for GCN, not a transductive
shortcut. A separately labeled fully-transductive variant (train also
inside the same static graph as test) is NOT used for the headline
comparison — see METHODOLOGY.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim_1: int = 64, hidden_dim_2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim_1, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim_1, hidden_dim_2, add_self_loops=True)
        self.classifier = nn.Linear(hidden_dim_2, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        h = F.relu(h)
        logits = self.classifier(h).squeeze(-1)
        return logits
