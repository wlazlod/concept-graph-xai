"""Concept-level beeswarm / strip plot (P1).

Per concept, distribution of summed signed SHAP across samples. Reveals both
direction (does the concept push predictions up or down?) and spread.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph


def concept_beeswarm(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    *,
    only_concepts: bool = True,
    sort_by_importance: bool = True,
    max_concepts: int | None = None,
    title: str | None = None,
    point_size: int = 4,
    point_opacity: float = 0.5,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a concept-level beeswarm of summed signed SHAP per sample.

    Each row is a concept (or feature, if ``only_concepts=False``); each point
    is one sample. The ``x`` value is the **sum** of SHAP across the concept's
    descendant features for that sample. Concept-level mean(|SHAP|) is shown
    as a bar overlay on the right axis when sorting is enabled.

    Parameters
    ----------
    graph:
        The ConceptGraph.
    feature_names:
        Names of the features matching the columns of ``shap_values``.
    shap_values:
        Per-sample SHAP values of shape ``(N, F)``.
    only_concepts:
        If True (default), drop feature leaves from the chart.
    sort_by_importance:
        If True (default), order rows by mean(|summed SHAP|) descending.
    max_concepts:
        Optionally cap the number of rows shown (top-K by importance).
    title:
        Figure title.
    point_size:
        Marker size for individual sample points.
    point_opacity:
        Marker opacity for individual sample points.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.

    Returns
    -------
    plotly.graph_objects.Figure
    """

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"shap_values must be 2D (N, F); got {arr.shape}")
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {arr.shape[1]} cols but feature_names has {len(feature_names)}"
        )

    name_to_idx: dict[str, int] = {n: i for i, n in enumerate(feature_names)}

    nodes: list[str] = []
    for node in graph.nodes_in_order():
        if node == graph.root:
            continue
        if only_concepts and graph.kind(node) == "feature":
            continue
        feats = [f for f in graph.descendant_features(node) if f in name_to_idx]
        if not feats:
            continue
        nodes.append(node)

    if not nodes:
        raise ValueError("no concepts (or features) match the SHAP feature names")

    summed: dict[str, np.ndarray] = {}
    importance: dict[str, float] = {}
    for node in nodes:
        idxs = [name_to_idx[f] for f in graph.descendant_features(node) if f in name_to_idx]
        block = arr[:, idxs]
        s = block.sum(axis=1)
        summed[node] = s
        importance[node] = float(np.abs(s).mean())

    if sort_by_importance:
        nodes = sorted(nodes, key=lambda n: importance[n], reverse=True)
    if max_concepts is not None:
        nodes = nodes[:max_concepts]

    fig = go.Figure()
    for node in nodes:
        s = summed[node]
        fig.add_trace(
            go.Box(
                x=s,
                y=[node] * len(s),
                name=node,
                orientation="h",
                boxpoints="all",
                jitter=0.5,
                pointpos=0,
                marker={"size": point_size, "opacity": point_opacity},
                line={"width": 1},
                fillcolor="rgba(0,0,0,0)",
                showlegend=False,
                hovertemplate="%{y}<br>SHAP: %{x:+.4f}<extra></extra>",
            )
        )

    fig.add_vline(x=0.0, line={"color": "black", "width": 1, "dash": "dash"})

    fig.update_layout(
        title=title or "Concept beeswarm — summed signed SHAP per sample",
        xaxis_title="summed signed SHAP",
        yaxis_title="concept",
        yaxis={"autorange": "reversed"},
        margin={"t": 60, "l": 160, "r": 30, "b": 60},
        height=max(300, 30 * len(nodes) + 120),
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
