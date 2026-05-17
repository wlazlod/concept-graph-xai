"""Sunburst colored by per-concept AUC loss (idea #3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import (
    build_sunburst_figure,
    hover_text,
    sunburst_layout,
)


def auc_drop_map(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    value: str = "auc_drop_mean",
    size: str = "feature_count",
    colorscale: str = "Reds",
    hide_root: bool = True,
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a sunburst where each concept is colored by its AUC drop.

    Sector area uses ``size`` (feature count by default), the colour intensity
    uses ``value`` (mean AUC drop by default). Set ``hide_root=False`` to
    keep the root sector visible.
    """

    if value not in df.columns:
        raise KeyError(f"{value!r} not in DataFrame; run metrics.auc_drop first")
    if size not in df.columns:
        raise KeyError(f"{size!r} not in DataFrame")

    arrays, ordered, sizes = sunburst_layout(graph, df, value=size, hide_root=hide_root)

    drop_vals = ordered[value].to_numpy(dtype=float)
    drop_for_color = np.where(np.isnan(drop_vals), 0.0, drop_vals)
    cmax = float(np.nanmax(np.abs(drop_vals))) if not np.all(np.isnan(drop_vals)) else 1.0
    cmin = -cmax if (np.nanmin(drop_vals) < 0) else 0.0

    hover_cols = [
        c
        for c in (
            value,
            "auc_drop_std",
            "ablated_score_mean",
            "baseline_score",
            "feature_count",
            "strategy",
        )
        if c in ordered.columns
    ]
    hover = hover_text(
        ordered,
        hover_cols,
        fmt={
            value: "+.4f",
            "auc_drop_std": ".4f",
            "ablated_score_mean": ".4f",
            "baseline_score": ".4f",
        },
    )

    return build_sunburst_figure(
        arrays,
        sizes,
        marker={
            "colors": drop_for_color,
            "colorscale": colorscale,
            "cmin": cmin,
            "cmax": cmax,
            "showscale": True,
            "colorbar": {"title": value},
        },
        hover=hover,
        title=title or "AUC drop per concept",
        layout_kwargs=layout_kwargs,
    )
