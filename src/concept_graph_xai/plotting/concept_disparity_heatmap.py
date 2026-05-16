"""Concept × protected-group SHAP disparity heatmap (P11)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import heatmap_color_kwargs

SortBy = Literal["max_abs", "depth"]


def concept_disparity_heatmap(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    only_concepts: bool = True,
    hide_root: bool = True,
    include_reference: bool = True,
    sort_by: SortBy | None = "max_abs",
    max_concepts: int | None = None,
    title: str | None = None,
    colorscale: str = "RdBu_r",
    layout_kwargs: dict[str, Any] | None = None,
    include_root: bool | None = None,
) -> go.Figure:
    """Render a concept × protected-group disparity heatmap.

    Diverging palette centred at 0 — the reference column is exactly
    zero, other columns are signed gaps. Positive gap = the model
    relies on this concept *more* for that group than for the reference;
    negative = less.

    Parameters
    ----------
    graph:
        The ConceptGraph (used for depth-aware ordering).
    df:
        Long-form output of :func:`concept_disparity`.
    only_concepts:
        If ``True`` (default), drop feature leaves from the chart.
    hide_root:
        If ``True`` (default), drop the root concept row. Renamed from
        the previous ``include_root`` flag for consistency with the
        sunburst family.
    include_reference:
        If ``True`` (default), keep the reference group's all-zero
        column as a visible baseline. Pass ``False`` to drop it for a
        tighter chart.
    sort_by:
        ``"max_abs"`` (default) orders concept rows by the maximum
        ``|gap|`` across groups, descending — surfaces the most
        disparate concepts. ``"depth"`` keeps graph DFS preorder.
        ``None`` keeps the DataFrame order.
    max_concepts:
        Optionally cap the number of rows shown (top-K by chosen
        ordering).
    title:
        Figure title. Defaults to ``"Concept SHAP disparity vs
        <reference>"`` using the reference label from ``df.attrs``.
    colorscale:
        Plotly diverging colorscale name. Default ``"RdBu_r"`` so a
        positive gap (the model over-relies on this concept for the
        protected group) renders red — same convention as
        ``concept_drift_sunburst``.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    if include_root is not None:
        import warnings

        warnings.warn(
            "include_root is deprecated; pass hide_root=not include_root instead",
            DeprecationWarning,
            stacklevel=2,
        )
        hide_root = not include_root

    if df.empty:
        raise ValueError("DataFrame is empty; run concept_disparity first")
    for col in ("name", "kind", "protected_group", "value"):
        if col not in df.columns:
            raise KeyError(f"required column {col!r} missing from DataFrame")

    work = df.copy()
    if only_concepts:
        work = work[work["kind"] == "concept"]
    if hide_root:
        work = work[work["name"] != graph.root]

    reference_group = df.attrs.get("reference_group")
    protected_order: list[str] = df.attrs.get("protected_order") or list(
        dict.fromkeys(work["protected_group"].astype(str))
    )
    if not include_reference and reference_group is not None:
        protected_order = [g for g in protected_order if g != reference_group]
        work = work[work["protected_group"] != reference_group]

    pivot = work.pivot_table(
        index="name", columns="protected_group", values="value", aggfunc="mean"
    )
    pivot = pivot.reindex(columns=[g for g in protected_order if g in pivot.columns])

    if sort_by == "max_abs":
        pivot = pivot.assign(_max=pivot.abs().max(axis=1)).sort_values(
            "_max", ascending=False
        ).drop(columns="_max")
    elif sort_by == "depth":
        dfs_order = [n for n in graph.nodes_in_order() if n in set(pivot.index)]
        pivot = pivot.reindex(index=dfs_order)
    elif sort_by is not None:
        raise ValueError(f"unknown sort_by={sort_by!r}; expected 'max_abs', 'depth', or None")

    if max_concepts is not None:
        pivot = pivot.head(max_concepts)

    if pivot.empty:
        raise ValueError("no rows to plot after filtering")

    agg = df.attrs.get("agg", "mean_abs")
    z = pivot.to_numpy(dtype=float)
    # Disparity is always signed — force diverging behaviour regardless of
    # df.attrs["agg"], because a magnitude-vs-magnitude gap can still go
    # negative (group < reference) and deserves a centred palette.
    color_kwargs = heatmap_color_kwargs(z, agg="mean_signed", colorscale=colorscale)

    auto_title = f"Concept SHAP disparity vs {reference_group}" if reference_group else "Concept SHAP disparity"
    auto_title += f" ({agg})"

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorbar={"title": f"{agg} gap"},
            hovertemplate="%{y} | %{x}<br>gap: %{z:+.4f}<extra></extra>",
            **color_kwargs,
        )
    )

    fig.update_layout(
        title=title or auto_title,
        xaxis={"side": "bottom", "tickangle": 0, "showgrid": False, "title": "protected group"},
        yaxis={"autorange": "reversed", "showgrid": False, "title": "concept"},
        margin={"t": 60, "l": 160, "r": 30, "b": 80},
        height=max(300, 28 * len(pivot) + 160),
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
