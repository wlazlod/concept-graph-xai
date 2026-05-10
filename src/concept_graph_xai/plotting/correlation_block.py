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

    # Depth = number of slashes in concept_path. Root has depth 0; top-level
    # concepts under root have depth 1; sub-concepts depth 2, etc. We stack
    # block labels in horizontal rows below the heatmap, with the deepest
    # concept closest to the heatmap and the top-level concepts furthest down.
    # This stops nested blocks (e.g. "Behaviour" + "Delinquency") from writing
    # their labels on top of each other.
    block_depths = [path.count("/") for path, _s, _e in result.blocks]
    skip_root = result.blocks and block_depths[0] == 0
    visible_depths = [d for d in block_depths if d > 0]
    max_depth = max(visible_depths) if visible_depths else 1
    row_height = 1.0
    label_top_y = -1.2  # closest to heatmap (deepest concepts)

    for (path, start, end), depth in zip(result.blocks, block_depths, strict=True):
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
        if show_block_labels and end - start >= 1 and depth >= 1:
            label = path.split("/")[-1]
            # Deeper concepts → smaller magnitude y (closer to heatmap).
            # Top-level branches (depth=1) → most negative y (further down).
            row = max_depth - depth  # 0 for deepest, max_depth-1 for top-level
            y_pos = label_top_y - row * row_height
            font_size = 12 if depth == 1 else max(8, 11 - (depth - 1))
            annotations.append(
                {
                    "x": (start + end - 1) / 2,
                    "y": y_pos,
                    "xref": "x",
                    "yref": "y",
                    "text": f"<b>{label}</b>" if depth == 1 else label,
                    "showarrow": False,
                    "font": {"size": font_size},
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

    label_band = max(1, max_depth) * row_height + 0.5  # space reserved below heatmap
    bottom_margin = int(60 + 22 * max_depth)
    _ = skip_root  # kept for clarity; root block has no label

    fig.update_layout(
        title=title,
        xaxis={"side": "bottom", "tickangle": 45, "showgrid": False, "range": [-0.5, n - 0.5]},
        yaxis={
            "autorange": "reversed",
            "showgrid": False,
            "range": [n - 0.5, label_top_y - label_band],
        },
        shapes=shapes,
        annotations=annotations,
        margin={"t": 40, "l": 40, "r": 40, "b": bottom_margin},
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
