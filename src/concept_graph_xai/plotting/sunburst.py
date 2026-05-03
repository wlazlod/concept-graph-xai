"""Generic sunburst plot driven by a ConceptGraph and a metric DataFrame."""

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


def sunburst(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    value: str = "count",
    title: str | None = None,
    colorscale: str | None = None,
    color_value: str | None = None,
    branchvalues: str = "total",
    extra_hover: list[str] | None = None,
    hover_fmt: dict[str, str] | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a sunburst from a ConceptGraph + a metric DataFrame.

    Parameters
    ----------
    graph:
        The ConceptGraph to render.
    df:
        Tidy DataFrame produced by one of the metric functions. Must be
        indexed by ``path`` and contain ``value`` (and ``color_value`` if
        coloring is requested).
    value:
        Column used for sector size. Defaults to ``"count"``.
    title:
        Figure title.
    colorscale:
        Plotly colorscale name (e.g. ``"Viridis"``, ``"Reds"``). When set,
        sectors are colored by ``color_value`` (which defaults to ``value``).
    color_value:
        Column used for color intensity. Defaults to ``value`` when
        ``colorscale`` is set.
    branchvalues:
        Plotly sunburst branchvalues (``"total"`` or ``"remainder"``).
    extra_hover:
        Additional columns to append to the hover tooltip.
    hover_fmt:
        Per-column ``format`` spec strings (e.g. ``{"importance_sum": ".4f"}``).
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    arrays = graph_to_arrays(graph)
    ordered = reindex_to_paths(df, arrays["ids"])
    if value not in ordered.columns:
        raise KeyError(f"value column {value!r} not in DataFrame; have {list(ordered.columns)}")

    values = ordered[value].fillna(0).to_numpy(dtype=float)

    marker: dict[str, Any] = {"line": {"width": 0.5, "color": "white"}}
    if colorscale is not None:
        cv = color_value or value
        if cv not in ordered.columns:
            raise KeyError(f"color_value column {cv!r} not in DataFrame")
        marker.update(
            colors=ordered[cv].fillna(0).to_numpy(dtype=float),
            colorscale=colorscale,
            showscale=True,
            cmid=0 if (ordered[cv].min() < 0 < ordered[cv].max()) else None,
            colorbar={"title": cv},
        )

    hover_columns = [value]
    for col in ("kind", "feature_count", "used_feature_count", "is_used"):
        if col in ordered.columns and col not in hover_columns:
            hover_columns.append(col)
    if extra_hover:
        for col in extra_hover:
            if col not in hover_columns:
                hover_columns.append(col)

    hover = hover_text(ordered, hover_columns, fmt=hover_fmt)

    fig = go.Figure(
        go.Sunburst(
            ids=arrays["ids"],
            labels=arrays["labels"],
            parents=arrays["parents"],
            values=values,
            branchvalues=branchvalues,
            marker=marker,
            hovertext=hover,
            hovertemplate="<b>%{label}</b><br>%{hovertext}<extra></extra>",
            insidetextorientation="radial",
        )
    )
    fig.update_layout(
        title=title,
        margin={"t": 40, "l": 0, "r": 0, "b": 0},
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
