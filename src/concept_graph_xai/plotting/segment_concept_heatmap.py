"""Segment × concept SHAP heatmap (P7)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph


def segment_concept_heatmap(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    only_concepts: bool = True,
    include_root: bool = False,
    sort_by: str | None = "max",
    max_concepts: int | None = None,
    title: str | None = None,
    colorscale: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a concept × segment SHAP heatmap.

    Parameters
    ----------
    graph:
        The ConceptGraph (used for depth-aware ordering).
    df:
        Long-form output of :func:`segment_importance`.
    only_concepts:
        If ``True`` (default), drop feature leaves.
    include_root:
        If ``False`` (default), drop the root concept row.
    sort_by:
        ``"max"`` (default) orders concept rows by the maximum value across
        segments, descending. ``"depth"`` keeps graph DFS preorder.
        ``None`` keeps the order in the DataFrame.
    max_concepts:
        Optionally cap the number of rows shown (top-K by chosen ordering).
    title:
        Figure title.
    colorscale:
        Override the colorscale. Defaults to ``"Reds"`` for ``mean_abs`` and
        a diverging ``"RdBu"`` (centred at 0) for ``mean_signed``.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    if df.empty:
        raise ValueError("DataFrame is empty; run segment_importance first")
    for col in ("name", "kind", "segment", "value"):
        if col not in df.columns:
            raise KeyError(f"required column {col!r} missing from DataFrame")

    work = df.copy()
    if only_concepts:
        work = work[work["kind"] == "concept"]
    if not include_root:
        work = work[work["name"] != graph.root]

    segment_order = df.attrs.get("segment_order") or list(
        dict.fromkeys(work["segment"].astype(str))
    )

    pivot = work.pivot_table(
        index="name", columns="segment", values="value", aggfunc="mean"
    )
    pivot = pivot.reindex(columns=[s for s in segment_order if s in pivot.columns])

    if sort_by == "max":
        pivot = pivot.assign(_max=pivot.max(axis=1)).sort_values(
            "_max", ascending=False
        ).drop(columns="_max")
    elif sort_by == "depth":
        dfs_order = [n for n in graph.nodes_in_order() if n in set(pivot.index)]
        pivot = pivot.reindex(index=dfs_order)
    elif sort_by is not None:
        raise ValueError(f"unknown sort_by={sort_by!r}; expected 'max', 'depth', or None")

    if max_concepts is not None:
        pivot = pivot.head(max_concepts)

    if pivot.empty:
        raise ValueError("no rows to plot after filtering")

    agg = df.attrs.get("agg", "mean_abs")
    z = pivot.to_numpy(dtype=float)
    if colorscale is None:
        colorscale = "RdBu" if agg == "mean_signed" else "Reds"

    heatmap_kwargs: dict[str, Any] = {
        "z": z,
        "x": list(pivot.columns),
        "y": list(pivot.index),
        "colorscale": colorscale,
        "colorbar": {"title": agg},
        "hovertemplate": "%{y} | %{x}<br>" + agg + ": %{z:.4f}<extra></extra>",
    }
    if agg == "mean_signed":
        cabs = float(np.nanmax(np.abs(z))) or 1e-9
        heatmap_kwargs.update(zmid=0.0, zmin=-cabs, zmax=cabs)
    else:
        heatmap_kwargs.update(zmin=0.0, zmax=float(np.nanmax(z)) or 1e-9)

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))

    fig.update_layout(
        title=title or f"Concept SHAP by segment ({agg})",
        xaxis={"side": "bottom", "tickangle": 0, "showgrid": False, "title": "segment"},
        yaxis={"autorange": "reversed", "showgrid": False, "title": "concept"},
        margin={"t": 60, "l": 160, "r": 30, "b": 80},
        height=max(300, 28 * len(pivot) + 160),
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
