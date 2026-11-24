"""GraphSAGE — the primary inductive model: SAGEConv aggregates a node's own
features plus its sampled/available neighbors' features via mean
aggregation, which generalizes naturally to nodes never seen during
training (held-out customers connected only to training-reference
neighbors, see src/graph/graph_builder.build_inductive_edges)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim_1: int = 64, hidden_dim_2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim_1)
        self.conv2 = SAGEConv(hidden_dim_1, hidden_dim_2)
        self.classifier = nn.Linear(hidden_dim_2, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        logits = self.classifier(h).squeeze(-1)
        return logits
