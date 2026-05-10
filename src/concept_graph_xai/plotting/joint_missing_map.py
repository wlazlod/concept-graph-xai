"""Sunburst overlay coloured by per-concept joint-missing rate (P13/P15b)."""

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


def joint_missing_map(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    value: str = "joint_missing_rate",
    size: str = "feature_count",
    colorscale: str = "Reds",
    hide_root: bool = True,
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a sunburst where each concept is coloured by its joint-missing rate.

    The DataFrame must come from :func:`joint_missing_rate`. Sector size uses
    ``feature_count`` so the shape matches the existing sunburst plots; colour
    intensity uses ``joint_missing_rate``. Set ``hide_root=False`` to keep
    the root sector visible.
    """

    if value not in df.columns:
        raise KeyError(f"{value!r} not in DataFrame; run joint_missing_rate first")
    if size not in df.columns:
        raise KeyError(f"{size!r} not in DataFrame")

    arrays = graph_to_arrays(graph, hide_root=hide_root)
    ordered = reindex_to_paths(df, arrays["ids"])

    sizes = ordered[size].fillna(0).to_numpy(dtype=float)
    rates = ordered[value].fillna(0).to_numpy(dtype=float)

    hover_cols = [value, "feature_count"]
    hover = hover_text(
        ordered,
        [c for c in hover_cols if c in ordered.columns],
        fmt={value: ".3f"},
    )

    fig = go.Figure(
        go.Sunburst(
            ids=arrays["ids"],
            labels=arrays["labels"],
            parents=arrays["parents"],
            values=sizes,
            branchvalues="total",
            marker={
                "colors": rates,
                "colorscale": colorscale,
                "cmin": 0.0,
                "cmax": max(1e-6, float(rates.max())),
                "showscale": True,
                "colorbar": {"title": value},
                "line": {"width": 0.5, "color": "white"},
            },
            hovertext=hover,
            hovertemplate="<b>%{label}</b><br>%{hovertext}<extra></extra>",
            insidetextorientation="radial",
        )
    )
    fig.update_layout(
        title=title or "Joint missingness per concept",
        margin={"t": 40, "l": 0, "r": 0, "b": 0},
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
