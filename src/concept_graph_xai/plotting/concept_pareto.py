"""Per-cohort Lorenz / Pareto curves of concept importance (P8)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import deprecated_kwarg_or

# Plotly qualitative palette, inlined so a Pareto chart's segment colours
# are independent of the branch palette (segments != tree branches).
_DEFAULT_SEGMENT_PALETTE: tuple[str, ...] = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
)


def concept_pareto(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    only_concepts: bool = True,
    hide_root: bool = True,
    segment_palette: Sequence[str] | None = None,
    show_equality_line: bool = True,
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
    include_root: bool | None = None,
) -> go.Figure:
    """Render per-segment Lorenz / Pareto curves of concept importance.

    For each cohort in the long-form DataFrame from
    :func:`segment_importance`, concepts are sorted by ``value`` descending
    and accumulated to produce a Lorenz curve:

    * x = ``(rank + 1) / n_concepts`` — cumulative share of concepts.
    * y = ``cum_value / total_value`` — cumulative share of importance.

    A curve hugging the dashed 45° equality line means the cohort's
    importance is spread evenly across concepts; a curve bulging up-left
    means a few concepts dominate that cohort's SHAP budget.

    Parameters
    ----------
    graph:
        ConceptGraph (used to drop the root row).
    df:
        Long-form DataFrame from :func:`segment_importance` — must contain
        ``name``, ``kind``, ``segment``, ``value``.
    only_concepts:
        If ``True`` (default), drop feature leaves before ranking.
    hide_root:
        If ``True`` (default), drop the root concept row (it aggregates
        every feature and would distort the curve). Renamed from
        ``include_root`` for consistency with the sunburst family.
    segment_palette:
        Custom palette for per-segment colours. Defaults to the Plotly
        qualitative palette.
    show_equality_line:
        Overlay the dashed 45° reference line. Default ``True``.
    title:
        Figure title.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    hide_root = deprecated_kwarg_or(
        include_root, hide_root, old="include_root", new="hide_root", transform=lambda v: not v
    )

    if df.empty:
        raise ValueError("DataFrame is empty; run segment_importance first")
    for col in ("name", "kind", "segment", "value"):
        if col not in df.columns:
            raise KeyError(f"required column {col!r} missing from DataFrame")

    work = df.copy()
    if only_concepts:
        work = work[work["kind"] == "concept"]
    if hide_root:
        work = work[work["name"] != graph.root]

    segment_order: list[str] = df.attrs.get("segment_order") or list(
        dict.fromkeys(work["segment"].astype(str))
    )

    palette: tuple[str, ...] = (
        tuple(segment_palette) if segment_palette else _DEFAULT_SEGMENT_PALETTE
    )
    if not palette:
        raise ValueError("segment_palette must contain at least one colour")

    fig = go.Figure()

    if show_equality_line:
        fig.add_trace(
            go.Scatter(
                x=[0.0, 1.0],
                y=[0.0, 1.0],
                mode="lines",
                line={"color": "rgba(0,0,0,0.4)", "dash": "dash", "width": 1},
                name="equality",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    plotted_segments = 0
    for color_idx, segment in enumerate(segment_order):
        block = work[work["segment"] == segment]
        if block.empty:
            continue
        values = block["value"].to_numpy(dtype=float)
        names = block["name"].tolist()
        if values.size == 0 or float(values.sum()) <= 0:
            continue
        order = np.argsort(-values)  # descending
        sorted_values = values[order]
        sorted_names = [names[i] for i in order]
        n = len(sorted_values)
        cum_share_x = np.arange(1, n + 1, dtype=float) / n
        cum_share_y = np.cumsum(sorted_values) / float(sorted_values.sum())
        # Prepend (0, 0) so each curve starts at the origin.
        xs = np.concatenate([[0.0], cum_share_x])
        ys = np.concatenate([[0.0], cum_share_y])
        labels_with_origin = ["—", *sorted_names]
        color = palette[color_idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 6, "color": color},
                name=str(segment),
                text=labels_with_origin,
                hovertemplate=(
                    f"{segment}<br>"
                    "rank %{x:.2%} of concepts<br>"
                    "captures %{y:.2%} of importance<br>"
                    "(next concept: %{text})<extra></extra>"
                ),
            )
        )
        plotted_segments += 1

    if plotted_segments == 0:
        raise ValueError("no segment had non-zero importance; nothing to plot")

    fig.update_layout(
        title=title or "Concept importance concentration per segment (Lorenz / Pareto)",
        xaxis={
            "title": "cumulative share of concepts (ranked by importance desc)",
            "range": [0.0, 1.05],
            "tickformat": ".0%",
        },
        yaxis={
            "title": "cumulative share of importance",
            "range": [0.0, 1.05],
            "tickformat": ".0%",
        },
        legend={"title": "segment"},
        margin={"t": 60, "l": 70, "r": 30, "b": 60},
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
