"""Concept drift delta sunburst (P10).

Sector area carries structural size (feature count, additive — so the
``branchvalues="total"`` invariant holds); sector colour carries the
per-concept ``delta`` between baseline and target periods with a
diverging palette centred at 0.
"""

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


def concept_drift_sunburst(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    value: str = "feature_count",
    delta_col: str = "delta",
    colorscale: str = "RdBu_r",
    hide_root: bool = True,
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a sunburst coloured by per-concept SHAP drift.

    Sector area uses ``value`` (default ``feature_count`` — additive, safe
    for ``branchvalues="total"``). Sector colour uses ``delta_col`` with a
    diverging palette centred at 0. With the default ``RdBu_r`` colorscale,
    *positive* deltas (importance grew between baseline and target) render
    *red* and negative deltas (importance shrank) render *blue*.

    Parameters
    ----------
    graph:
        The ConceptGraph.
    df:
        DataFrame from :func:`concept_drift_delta` — must contain
        ``feature_count`` (or whatever ``value`` is set to) plus
        ``baseline``, ``target``, ``delta``.
    value:
        Column used for sector area. Default ``feature_count`` keeps the
        plot honest under ``branchvalues="total"``.
    delta_col:
        Column used for sector colour. Default ``delta``.
    colorscale:
        Plotly diverging colorscale name. Default ``RdBu_r`` (positive =
        red, negative = blue).
    hide_root:
        Drop the root concept by default (consistent with the other
        sunburst plots).
    title:
        Figure title.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    for required in (value, delta_col):
        if required not in df.columns:
            raise KeyError(f"required column {required!r} missing from DataFrame")

    arrays, ordered, sizes = sunburst_layout(graph, df, value=value, hide_root=hide_root)

    delta_vals = ordered[delta_col].to_numpy(dtype=float)
    if np.all(np.isnan(delta_vals)):
        raise ValueError(
            f"all values in {delta_col!r} are NaN; nothing to colour. "
            "Pass at least one period with a defined value."
        )
    delta_for_color = np.where(np.isnan(delta_vals), 0.0, delta_vals)
    cmax = float(np.nanmax(np.abs(delta_vals))) or 1e-9

    hover_cols = [c for c in ("baseline", "target", delta_col, value) if c in ordered.columns]
    hover = hover_text(
        ordered,
        hover_cols,
        fmt={"baseline": ".4f", "target": ".4f", delta_col: "+.4f"},
    )

    baseline_label = df.attrs.get("baseline_period")
    target_label = df.attrs.get("target_period")
    auto_title = "Concept SHAP drift"
    if baseline_label and target_label:
        auto_title += f" — {baseline_label} -> {target_label}"

    return build_sunburst_figure(
        arrays,
        sizes,
        marker={
            "colors": delta_for_color,
            "colorscale": colorscale,
            "cmid": 0.0,
            "cmin": -cmax,
            "cmax": cmax,
            "showscale": True,
            "colorbar": {"title": delta_col},
        },
        hover=hover,
        title=title or auto_title,
        layout_kwargs=layout_kwargs,
    )
