"""Sunburst overlay coloured by per-concept joint-missing rate (P13/P15b)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import (
    build_sunburst_figure,
    hover_text,
    sunburst_layout,
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

    arrays, ordered, sizes = sunburst_layout(graph, df, value=size, hide_root=hide_root)
    rates = ordered[value].fillna(0).to_numpy(dtype=float)
    cmax = max(1e-6, float(rates.max()))

    hover_cols = [c for c in (value, "feature_count") if c in ordered.columns]
    hover = hover_text(ordered, hover_cols, fmt={value: ".3f"})

    return build_sunburst_figure(
        arrays,
        sizes,
        marker={
            "colors": rates,
            "colorscale": colorscale,
            "cmin": 0.0,
            "cmax": cmax,
            "showscale": True,
            "colorbar": {"title": value},
        },
        hover=hover,
        title=title or "Joint missingness per concept",
        layout_kwargs=layout_kwargs,
    )
