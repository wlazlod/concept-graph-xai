"""Sunburst variant where unused branches are rendered grey (idea #2)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import (
    graph_to_arrays,
    hover_text,
    reindex_to_paths,
)


def utilization_map(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    value: str = "feature_count",
    used_color: str = "#1f77b4",
    unused_color: str = "#d3d3d3",
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a sunburst where unused branches are grey.

    The DataFrame must be the output of
    :func:`concept_graph_xai.metrics.utilization` (it requires the ``is_used``
    column).
    """

    if "is_used" not in df.columns:
        raise KeyError("utilization_map expects DataFrame from metrics.utilization (no is_used col)")

    arrays = graph_to_arrays(graph)
    ordered = reindex_to_paths(df, arrays["ids"])

    if value not in ordered.columns:
        raise KeyError(f"value column {value!r} not in DataFrame")
    values = ordered[value].fillna(0).to_numpy(dtype=float)

    colors = [used_color if bool(u) else unused_color for u in ordered["is_used"].to_numpy()]

    hover_cols = [value, "is_used", "used_feature_count", "feature_count", "importance_sum"]
    hover_cols = [c for c in hover_cols if c in ordered.columns]
    hover = hover_text(
        ordered,
        hover_cols,
        fmt={"importance_sum": ".4f"},
    )

    fig = go.Figure(
        go.Sunburst(
            ids=arrays["ids"],
            labels=arrays["labels"],
            parents=arrays["parents"],
            values=values,
            branchvalues="total",
            marker={"colors": colors, "line": {"width": 0.5, "color": "white"}},
            hovertext=hover,
            hovertemplate="<b>%{label}</b><br>%{hovertext}<extra></extra>",
            insidetextorientation="radial",
        )
    )
    fig.update_layout(
        title=title or "Concept utilization (grey = unused)",
        margin={"t": 40, "l": 0, "r": 0, "b": 0},
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
