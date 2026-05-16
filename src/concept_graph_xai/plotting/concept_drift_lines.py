"""Multi-period concept SHAP drift lines (P9)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import branch_colors


def concept_drift_lines(
    graph: ConceptGraph,
    df: pd.DataFrame,
    *,
    only_concepts: bool = True,
    include_root: bool = False,
    top_k: int | None = 10,
    branch_palette: Sequence[str] | None = None,
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render one line per concept across periods.

    Parameters
    ----------
    graph:
        The ConceptGraph (used for branch-hierarchical colouring).
    df:
        Long-form DataFrame from :func:`attribution_drift`.
    only_concepts:
        If ``True`` (default), drop feature leaves.
    include_root:
        If ``False`` (default), drop the root concept row.
    top_k:
        If set, keep only the K concepts with the highest
        max-across-periods value (default 10) to avoid spaghetti charts.
        Pass ``None`` to show every concept.
    branch_palette:
        Custom palette for branch base hues. Defaults to the Plotly
        qualitative palette.
    title:
        Figure title.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    if df.empty:
        raise ValueError("DataFrame is empty; run attribution_drift first")
    for col in ("name", "kind", "period", "value"):
        if col not in df.columns:
            raise KeyError(f"required column {col!r} missing from DataFrame")

    work = df.copy()
    if only_concepts:
        work = work[work["kind"] == "concept"]
    if not include_root:
        work = work[work["name"] != graph.root]

    period_order: list[str] = df.attrs.get("period_order") or list(
        dict.fromkeys(work["period"].astype(str))
    )

    pivot = work.pivot_table(
        index="name", columns="period", values="value", aggfunc="mean"
    )
    pivot = pivot.reindex(columns=[p for p in period_order if p in pivot.columns])

    if top_k is not None and top_k > 0:
        max_per_concept = pivot.max(axis=1, skipna=True)
        ordering = max_per_concept.sort_values(ascending=False).head(top_k).index
        pivot = pivot.loc[ordering]
    else:
        # Sort by max-across-periods desc anyway so the legend lists the
        # most prominent concepts first.
        max_per_concept = pivot.max(axis=1, skipna=True)
        pivot = pivot.loc[max_per_concept.sort_values(ascending=False).index]

    if pivot.empty:
        raise ValueError("no concept rows to plot after filtering")

    concept_ids = ["/".join(graph.path(name)) for name in pivot.index]
    colors = branch_colors(graph, concept_ids, palette=branch_palette)

    agg = df.attrs.get("agg", "mean_abs")

    fig = go.Figure()
    for (name, row), color in zip(pivot.iterrows(), colors, strict=True):
        fig.add_trace(
            go.Scatter(
                x=list(pivot.columns),
                y=row.to_numpy(dtype=float),
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"size": 7, "color": color},
                name=str(name),
                hovertemplate=(
                    f"{name}<br>period: %{{x}}<br>{agg}: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    suffix = f" — top {top_k} concepts" if top_k is not None and top_k < len(pivot) else ""
    fig.update_layout(
        title=title or f"Concept SHAP drift across periods ({agg}){suffix}",
        xaxis={"title": "period", "type": "category", "categoryorder": "array", "categoryarray": list(pivot.columns)},
        yaxis={"title": agg},
        legend={"title": "concept"},
        margin={"t": 60, "l": 70, "r": 30, "b": 60},
        height=max(360, 30 * len(pivot) + 200),
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
