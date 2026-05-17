"""Generic sunburst plot driven by a ConceptGraph and a metric DataFrame."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import (
    branch_colors,
    build_sunburst_figure,
    hover_text,
    sunburst_layout,
)

ColorBy = Literal["auto", "branch", "value", "none"]


def sunburst(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    value: str = "count",
    title: str | None = None,
    colorscale: str | None = None,
    color_value: str | None = None,
    color_by: ColorBy = "auto",
    branch_palette: Sequence[str] | None = None,
    hide_root: bool = True,
    branchvalues: str = "total",
    extra_hover: list[str] | None = None,
    hover_fmt: dict[str, str] | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a sunburst from a ConceptGraph + a metric DataFrame.

    Parameters
    ----------
    graph:
        The ConceptGraph to render.
    df:
        Tidy DataFrame produced by one of the metric functions. Must be
        indexed by ``path`` and contain ``value`` (and ``color_value`` if
        coloring is requested).
    value:
        Column used for sector size. Defaults to ``"count"``.
    title:
        Figure title.
    colorscale:
        Plotly colorscale name (e.g. ``"Viridis"``, ``"Reds"``). When set,
        sectors are colored by ``color_value`` (which defaults to ``value``).
    color_value:
        Column used for color intensity. Defaults to ``value`` when
        ``colorscale`` is set.
    color_by:
        How to colour sectors. ``"auto"`` (default) picks ``"value"`` when a
        ``colorscale`` is given and ``"branch"`` otherwise. ``"branch"``
        forces categorical-per-top-level-branch colouring (using
        ``branch_palette``). ``"value"`` forces colorscale-based colouring
        (raises if ``colorscale`` is not given). ``"none"`` disables per-sector
        colour overrides (raw Plotly defaults).
    branch_palette:
        CSS color sequence used when colouring by branch. Defaults to the
        Plotly qualitative palette.
    hide_root:
        When ``True`` (default) the root concept is omitted and its direct
        children form the centre ring. Pass ``False`` to keep the legacy
        rendering with the root sector visible.
    branchvalues:
        Plotly sunburst branchvalues (``"total"`` or ``"remainder"``).
    extra_hover:
        Additional columns to append to the hover tooltip.
    hover_fmt:
        Per-column ``format`` spec strings (e.g. ``{"importance_sum": ".4f"}``).
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    arrays, ordered, sizes = sunburst_layout(graph, df, value=value, hide_root=hide_root)

    resolved = _resolve_color_by(color_by, colorscale)
    marker: dict[str, Any] = {}
    if resolved == "value":
        cv = color_value or value
        if cv not in ordered.columns:
            raise KeyError(f"color_value column {cv!r} not in DataFrame")
        cv_values = ordered[cv].fillna(0).to_numpy(dtype=float)
        cv_min = float(ordered[cv].min())
        cv_max = float(ordered[cv].max())
        marker.update(
            colors=cv_values,
            colorscale=colorscale,
            showscale=True,
            cmid=0 if (cv_min < 0 < cv_max) else None,
            colorbar={"title": cv},
        )
    elif resolved == "branch":
        marker["colors"] = branch_colors(graph, arrays["ids"], palette=branch_palette)

    hover_columns = [value]
    for col in ("kind", "feature_count", "used_feature_count", "is_used"):
        if col in ordered.columns and col not in hover_columns:
            hover_columns.append(col)
    if extra_hover:
        for col in extra_hover:
            if col not in hover_columns:
                hover_columns.append(col)

    hover = hover_text(ordered, hover_columns, fmt=hover_fmt)

    return build_sunburst_figure(
        arrays,
        sizes,
        marker=marker,
        hover=hover,
        title=title,
        branchvalues=branchvalues,
        layout_kwargs=layout_kwargs,
    )


def _resolve_color_by(
    color_by: ColorBy, colorscale: str | None
) -> Literal["branch", "value", "none"]:
    if color_by == "auto":
        return "value" if colorscale is not None else "branch"
    if color_by == "value":
        if colorscale is None:
            raise ValueError("color_by='value' requires a colorscale (e.g. colorscale='Viridis')")
        return "value"
    if color_by == "branch":
        if colorscale is not None:
            warnings.warn("color_by='branch' ignores the supplied colorscale", stacklevel=3)
        return "branch"
    if color_by == "none":
        return "none"
    raise ValueError(f"unknown color_by={color_by!r}")
