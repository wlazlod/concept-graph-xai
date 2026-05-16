"""Signed concept SHAP bar chart with bootstrap CI error bars (P2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import branch_colors


def signed_concept_bar(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    only_concepts: bool = True,
    sort: bool = True,
    max_concepts: int | None = None,
    value: str | None = None,
    branch_palette: Sequence[str] | None = None,
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a horizontal bar chart of per-concept signed SHAP with CI bars.

    The DataFrame must come from :func:`bootstrap_importance` (it requires
    ``ci_lo`` / ``ci_hi`` columns and one of ``mean_signed_shap`` /
    ``mean_abs_shap``).

    Parameters
    ----------
    only_concepts:
        If ``True`` (default), drop feature leaves and the root from the chart.
    sort:
        If ``True`` (default), order bars by ``|mean|`` descending.
    max_concepts:
        Optionally cap the number of bars (top-K by ``|mean|``).
    value:
        Override the column carrying the bar value. Defaults to whichever of
        ``mean_signed_shap`` / ``mean_abs_shap`` is present.
    branch_palette:
        Custom palette for branch base hues. Defaults to the Plotly qualitative
        palette.
    title:
        Figure title.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    if value is None:
        for candidate in ("mean_signed_shap", "mean_abs_shap"):
            if candidate in df.columns:
                value = candidate
                break
        if value is None:
            raise KeyError("no value column found; expected 'mean_signed_shap' or 'mean_abs_shap'")
    for required in (value, "ci_lo", "ci_hi"):
        if required not in df.columns:
            raise KeyError(f"required column {required!r} missing from DataFrame")

    work = df.copy()
    if only_concepts:
        work = work[(work["kind"] == "concept") & (work["name"] != graph.root)]
    if sort:
        work = (
            work.assign(_abs=work[value].abs())
            .sort_values("_abs", ascending=False)
            .drop(columns="_abs")
        )
    if max_concepts is not None:
        work = work.head(max_concepts)

    ids = ["/".join(graph.path(name)) for name in work["name"]]
    colors = branch_colors(graph, ids, palette=branch_palette)

    means = work[value].to_numpy(dtype=float)
    lo = work["ci_lo"].to_numpy(dtype=float)
    hi = work["ci_hi"].to_numpy(dtype=float)
    err_minus = means - lo
    err_plus = hi - means

    fig = go.Figure(
        go.Bar(
            x=means,
            y=work["name"].tolist(),
            orientation="h",
            marker={"color": colors, "line": {"color": "rgba(0,0,0,0.4)", "width": 0.5}},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": err_plus,
                "arrayminus": err_minus,
                "color": "rgba(0,0,0,0.6)",
                "thickness": 1.5,
                "width": 5,
            },
            customdata=np.stack([lo, hi], axis=1),
            hovertemplate=(
                "%{y}<br>"
                f"{value}: %{{x:+.4f}}<br>"
                "CI: [%{customdata[0]:+.4f}, %{customdata[1]:+.4f}]<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0.0, line={"color": "black", "width": 1, "dash": "dash"})

    ci_pct = df.attrs.get("ci")
    suffix = f" — {round(ci_pct * 100)}% CI" if ci_pct is not None else ""
    fig.update_layout(
        title=title or f"Signed concept SHAP{suffix}",
        xaxis_title=value,
        yaxis_title="concept",
        yaxis={"autorange": "reversed"},
        margin={"t": 60, "l": 160, "r": 30, "b": 60},
        height=max(300, 30 * len(work) + 120),
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
