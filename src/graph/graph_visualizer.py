"""Sampled-subgraph visualization for the dashboard.

Never renders the full ~42k-node graph in the browser — always samples a
small, connected-ish neighborhood around a seed set of customers.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from src.graph.graph_builder import GraphBundle


def sample_subgraph(graph: GraphBundle, seed_nodes: list[int], max_nodes: int = 150) -> nx.Graph:
    edge_index = graph.edge_index.numpy()
    g = nx.Graph()
    g.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    nodes = set(seed_nodes)
    frontier = set(seed_nodes)
    while len(nodes) < max_nodes and frontier:
        next_frontier = set()
        for n in frontier:
            if n in g:
                next_frontier.update(g.neighbors(n))
        next_frontier -= nodes
        if not next_frontier:
            break
        remaining = max_nodes - len(nodes)
        take = list(next_frontier)[:remaining]
        nodes.update(take)
        frontier = set(take)

    return g.subgraph(nodes).copy()


def plot_subgraph(
    sub_g: nx.Graph,
    node_color_values: dict[int, float],
    node_labels: dict[int, str] | None = None,
    color_label: str = "Risk score",
) -> go.Figure:
    if sub_g.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No graph data available for this selection.",
                            showarrow=False, x=0.5, y=0.5)
        return fig

    pos = nx.spring_layout(sub_g, seed=42, k=0.6)
    node_labels = node_labels or {}

    edge_x, edge_y = [], []
    for u, v in sub_g.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=0.6, color="rgba(150,150,150,0.5)"),
                             hoverinfo="none")

    node_ids = list(sub_g.nodes())
    node_x = [pos[n][0] for n in node_ids]
    node_y = [pos[n][1] for n in node_ids]
    colors = [node_color_values.get(n, 0.0) for n in node_ids]
    text = [node_labels.get(n, str(n)) for n in node_ids]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text", text=text,
        marker=dict(size=10, color=colors, colorscale="RdYlGn_r", showscale=True,
                     colorbar=dict(title=color_label), line=dict(width=0.5, color="white")),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    return fig
