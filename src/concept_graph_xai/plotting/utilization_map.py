"""Sunburst variant where unused branches are rendered grey (idea #2).

Used sectors default to a hierarchical per-branch palette: each top-level
concept gets a base hue, sub-concepts and their leaves get progressively
lighter shades of that hue. Pass ``used_color="<css>"`` for a single solid
colour (the legacy look).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import (
    branch_colors,
    graph_to_arrays,
    hover_text,
    reindex_to_paths,
)


def utilization_map(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    value: str = "feature_count",
    used_color: str | None = None,
    unused_color: str = "#d3d3d3",
    branch_palette: Sequence[str] | None = None,
    hide_root: bool = True,
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a sunburst where unused branches are grey.

    The DataFrame must be the output of
    :func:`concept_graph_xai.metrics.utilization` (it requires the ``is_used``
    column). By default sector area encodes ``feature_count`` and colour
    encodes both branch identity (hue) and is-used status (grey when not used)
    — the chart subsumes the standalone ``sunburst(..., feature_counts(...))``
    structural view.

    Parameters
    ----------
    used_color:
        If ``None`` (default), used sectors are coloured by their top-level
        branch with hierarchical shading (sub-concepts get lighter shades of
        the branch hue). Pass a CSS colour to fall back to a single solid
        colour for every used sector (legacy behaviour).
    branch_palette:
        Custom palette for branch base hues. Defaults to the Plotly
        qualitative palette.
    hide_root:
        When ``True`` (default) the root concept is omitted; pass ``False``
        to keep the legacy root sector.
    """

    if "is_used" not in df.columns:
        raise KeyError("utilization_map expects DataFrame from metrics.utilization (no is_used col)")

    arrays = graph_to_arrays(graph, hide_root=hide_root)
    ordered = reindex_to_paths(df, arrays["ids"])

    if value not in ordered.columns:
        raise KeyError(f"value column {value!r} not in DataFrame")
    values = ordered[value].fillna(0).to_numpy(dtype=float)

    is_used = ordered["is_used"].to_numpy()
    if used_color is None:
        used_palette = branch_colors(graph, arrays["ids"], palette=branch_palette)
        colors = [used_palette[i] if bool(u) else unused_color for i, u in enumerate(is_used)]
    else:
        colors = [used_color if bool(u) else unused_color for u in is_used]

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
