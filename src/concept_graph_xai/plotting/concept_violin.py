"""Concept-level violin plot of summed signed SHAP per sample (P1).

One horizontal violin per concept. The width at each x is the kernel-density
estimate of the per-sample summed SHAP across descendants. Compared to a
strip / beeswarm plot, the violin shape conveys distribution shape (skew,
multimodality) at a glance without the visual clutter of many points.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import branch_colors

ViolinPoints = Literal[False, "outliers", "all", "suspectedoutliers"]


def concept_violin(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    *,
    only_concepts: bool = True,
    sort_by_importance: bool = True,
    max_concepts: int | None = None,
    title: str | None = None,
    points: ViolinPoints = "outliers",
    box_visible: bool = True,
    meanline_visible: bool = True,
    branch_palette: Sequence[str] | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a per-concept violin of summed signed SHAP across samples.

    Each row is a concept (or feature, if ``only_concepts=False``); the violin
    shape is the KDE of the per-sample sum of SHAP across the concept's
    descendant features. A central box-and-whiskers and a mean line can be
    overlaid via ``box_visible`` and ``meanline_visible``.

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
    points:
        Which raw points to overlay on the violin. ``"outliers"`` (default)
        shows only points beyond the whiskers; ``"all"`` shows every sample
        (slow for large N); ``False`` shows none.
    box_visible:
        Draw the box-and-whiskers inside each violin. Default ``True``.
    meanline_visible:
        Draw the mean as a dashed line inside each violin. Default ``True``.
    branch_palette:
        Custom palette for branch base hues (each violin is tinted by its
        top-level branch with hierarchical shading). Defaults to the Plotly
        qualitative palette.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
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

    node_ids = ["/".join(graph.path(n)) for n in nodes]
    colors = branch_colors(graph, node_ids, palette=branch_palette)

    fig = go.Figure()
    for node, color in zip(nodes, colors, strict=True):
        s = summed[node]
        fig.add_trace(
            go.Violin(
                x=s,
                y=[node] * len(s),
                name=node,
                orientation="h",
                points=points,
                box_visible=box_visible,
                meanline_visible=meanline_visible,
                fillcolor=color,
                line={"color": "rgba(0,0,0,0.6)", "width": 1},
                opacity=0.85,
                marker={"size": 3, "opacity": 0.5, "color": "rgba(0,0,0,0.5)"},
                showlegend=False,
                hoveron="violins+points",
                hovertemplate="%{y}<br>SHAP: %{x:+.4f}<extra></extra>",
                spanmode="hard",
            )
        )

    fig.add_vline(x=0.0, line={"color": "black", "width": 1, "dash": "dash"})

    fig.update_layout(
        title=title or "Concept SHAP distribution (violin)",
        xaxis_title="summed signed SHAP",
        yaxis_title="concept",
        yaxis={"autorange": "reversed"},
        violingap=0.3,
        violinmode="overlay",
        margin={"t": 60, "l": 160, "r": 30, "b": 60},
        height=max(300, 50 * len(nodes) + 120),
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
