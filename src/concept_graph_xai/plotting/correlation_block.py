"""Block-structured correlation heatmap (P14, P15a, P17)."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from concept_graph_xai.metrics.correlation import CorrelationResult


def correlation_block(
    result: CorrelationResult,
    *,
    title: str | None = None,
    show_block_labels: bool = True,
    annotate_mean_abs: bool = True,
    colorscale: str = "RdBu",
    zmid: float = 0.0,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a correlation matrix with concept-block separators.

    Works on the output of any of :func:`feature_correlation`,
    :func:`nullity_correlation`, or :func:`shap_correlation` — they all return
    a :class:`CorrelationResult`.

    Parameters
    ----------
    result:
        Output of one of the correlation metrics.
    title:
        Figure title.
    show_block_labels:
        Draw the concept name above each diagonal block.
    annotate_mean_abs:
        Print ``mean(|r|)`` inside each diagonal block.
    colorscale:
        Plotly colorscale name. Default ``RdBu`` is symmetric around zero.
    zmid:
        Mid value for the colorscale. Use ``0`` for a diverging palette.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    matrix = result.matrix
    n = matrix.shape[0]

    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale=colorscale,
            zmid=zmid,
            zmin=-1.0,
            zmax=1.0,
            colorbar={"title": f"{result.method} ρ"},
            hovertemplate="%{x} ↔ %{y}<br>%{z:.3f}<extra></extra>",
        )
    )

    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    stats_lookup = result.block_stats.set_index("concept_path").to_dict("index")

    for path, start, end in result.blocks:
        # Diagonal block border
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": start - 0.5,
                "x1": end - 0.5,
                "y0": start - 0.5,
                "y1": end - 0.5,
                "line": {"color": "black", "width": 1.5},
                "fillcolor": "rgba(0,0,0,0)",
            }
        )
        if show_block_labels and end - start >= 1:
            label = path.split("/")[-1]
            annotations.append(
                {
                    "x": (start + end - 1) / 2,
                    "y": -1.2,
                    "xref": "x",
                    "yref": "y",
                    "text": f"<b>{label}</b>",
                    "showarrow": False,
                    "font": {"size": 11},
                }
            )
        if annotate_mean_abs and end - start >= 2:
            stats = stats_lookup.get(path, {})
            mean_abs = stats.get("mean_abs")
            if mean_abs is not None:
                annotations.append(
                    {
                        "x": (start + end - 1) / 2,
                        "y": (start + end - 1) / 2,
                        "xref": "x",
                        "yref": "y",
                        "text": f"|ρ̄|={mean_abs:.2f}",
                        "showarrow": False,
                        "font": {"size": 10, "color": "black"},
                        "bgcolor": "rgba(255,255,255,0.6)",
                    }
                )

    fig.update_layout(
        title=title,
        xaxis={"side": "bottom", "tickangle": 45, "showgrid": False, "range": [-0.5, n - 0.5]},
        yaxis={"autorange": "reversed", "showgrid": False, "range": [n - 0.5, -1.5]},
        shapes=shapes,
        annotations=annotations,
        margin={"t": 40, "l": 40, "r": 40, "b": 80},
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
