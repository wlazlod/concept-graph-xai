"""Sunburst colored by per-concept AUC loss (idea #3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import (
    graph_to_arrays,
    hover_text,
    reindex_to_paths,
)


def auc_drop_map(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    value: str = "auc_drop_mean",
    size: str = "feature_count",
    colorscale: str = "Reds",
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a sunburst where each concept is colored by its AUC drop.

    Sector area uses ``size`` (feature count by default), the colour intensity
    uses ``value`` (mean AUC drop by default).
    """

    if value not in df.columns:
        raise KeyError(f"{value!r} not in DataFrame; run metrics.auc_drop first")
    if size not in df.columns:
        raise KeyError(f"{size!r} not in DataFrame")

    arrays = graph_to_arrays(graph)
    ordered = reindex_to_paths(df, arrays["ids"])

    sizes = ordered[size].fillna(0).to_numpy(dtype=float)
    drop_vals = ordered[value].to_numpy(dtype=float)
    drop_for_color = np.where(np.isnan(drop_vals), 0.0, drop_vals)

    cmax = float(np.nanmax(np.abs(drop_vals))) if not np.all(np.isnan(drop_vals)) else 1.0
    cmin = -cmax if (np.nanmin(drop_vals) < 0) else 0.0

    hover_cols = [
        value,
        "auc_drop_std",
        "ablated_score_mean",
        "baseline_score",
        "feature_count",
        "strategy",
    ]
    hover_cols = [c for c in hover_cols if c in ordered.columns]
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

    fig = go.Figure(
        go.Sunburst(
            ids=arrays["ids"],
            labels=arrays["labels"],
            parents=arrays["parents"],
            values=sizes,
            branchvalues="total",
            marker={
                "colors": drop_for_color,
                "colorscale": colorscale,
                "cmin": cmin,
                "cmax": cmax,
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
        title=title or "AUC drop per concept",
        margin={"t": 40, "l": 0, "r": 0, "b": 0},
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
