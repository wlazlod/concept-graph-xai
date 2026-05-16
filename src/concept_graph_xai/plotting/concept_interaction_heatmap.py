"""Concept × concept SHAP interaction heatmap (P3)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.plotting._layout import heatmap_color_kwargs


def concept_interaction_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str | None = None,
    show_diagonal_box: bool = True,
    annotate_top_k: int | None = 5,
    colorscale: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a concept × concept SHAP-interaction heatmap.

    Parameters
    ----------
    matrix:
        Square DataFrame returned by :func:`concept_interaction_matrix`.
    title:
        Figure title.
    show_diagonal_box:
        If ``True`` (default), draw a faint box around the main diagonal so
        within-concept self-interaction is visually separated from cross-pair
        interactions.
    annotate_top_k:
        Annotate the largest ``k`` off-diagonal cells with their value
        (default 5). Pass ``None`` to disable.
    colorscale:
        Override the colorscale. Defaults to ``"Reds"`` for ``mean_abs`` and a
        diverging ``"RdBu"`` (centred at 0) for ``mean_signed``.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square; got {matrix.shape}")
    if list(matrix.index) != list(matrix.columns):
        raise ValueError("matrix rows and columns must be identical")

    agg = matrix.attrs.get("agg", "mean_abs")
    z = matrix.to_numpy(dtype=float)
    n = z.shape[0]
    labels = list(matrix.index)

    value_fmt = "+.4f" if agg == "mean_signed" else ".4f"
    heatmap_kwargs: dict[str, Any] = {
        "z": z,
        "x": labels,
        "y": labels,
        "colorbar": {"title": agg},
        "hovertemplate": "%{y} ↔ %{x}<br>" + agg + ": %{z:" + value_fmt + "}<extra></extra>",
        **heatmap_color_kwargs(z, agg=agg, colorscale=colorscale),
    }

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))

    shapes: list[dict[str, Any]] = []
    if show_diagonal_box:
        for i in range(n):
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": i - 0.5,
                    "x1": i + 0.5,
                    "y0": i - 0.5,
                    "y1": i + 0.5,
                    "line": {"color": "rgba(0,0,0,0.4)", "width": 1.0},
                    "fillcolor": "rgba(0,0,0,0)",
                }
            )

    annotations: list[dict[str, Any]] = []
    if annotate_top_k is not None and annotate_top_k > 0:
        # Off-diagonal cells in the upper triangle (matrix is symmetric)
        candidates: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                candidates.append((float(z[i, j]), i, j))
        candidates.sort(key=lambda t: abs(t[0]), reverse=True)
        for value, i, j in candidates[:annotate_top_k]:
            text = f"{value:+.3f}" if agg == "mean_signed" else f"{value:.3f}"
            annotations.append(
                {
                    "x": j,
                    "y": i,
                    "xref": "x",
                    "yref": "y",
                    "text": text,
                    "showarrow": False,
                    "font": {"size": 10, "color": "black"},
                    "bgcolor": "rgba(255,255,255,0.65)",
                }
            )

    fig.update_layout(
        title=title or f"Concept × concept SHAP interaction ({agg})",
        xaxis={"side": "bottom", "tickangle": 45, "showgrid": False, "constrain": "domain"},
        yaxis={
            "autorange": "reversed",
            "showgrid": False,
            "scaleanchor": "x",
            "constrain": "domain",
        },
        shapes=shapes,
        annotations=annotations,
        margin={"t": 60, "l": 140, "r": 30, "b": 120},
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
