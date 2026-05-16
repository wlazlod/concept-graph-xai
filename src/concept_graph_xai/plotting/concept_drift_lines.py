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
    hide_root: bool = True,
    max_concepts: int | None = None,
    branch_palette: Sequence[str] | None = None,
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
    include_root: bool | None = None,
    top_k: int | None = None,
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
    hide_root:
        If ``True`` (default), drop the root concept row. Renamed from
        ``include_root`` for consistency with the sunburst family.
    max_concepts:
        If set, keep only the K concepts with the highest
        max-across-periods value to avoid spaghetti charts. Default
        ``None`` shows every concept. Renamed from ``top_k`` for
        consistency with the rest of the library; ``top_k`` is still
        accepted as a deprecated alias.
    branch_palette:
        Custom palette for branch base hues. Defaults to the Plotly
        qualitative palette.
    title:
        Figure title.
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
    if top_k is not None:
        import warnings

        warnings.warn(
            "top_k is deprecated; pass max_concepts instead",
            DeprecationWarning,
            stacklevel=2,
        )
        max_concepts = top_k

    if df.empty:
        raise ValueError("DataFrame is empty; run attribution_drift first")
    for col in ("name", "kind", "period", "value"):
        if col not in df.columns:
            raise KeyError(f"required column {col!r} missing from DataFrame")

    work = df.copy()
    if only_concepts:
        work = work[work["kind"] == "concept"]
    if hide_root:
        work = work[work["name"] != graph.root]

    period_order: list[str] = df.attrs.get("period_order") or list(
        dict.fromkeys(work["period"].astype(str))
    )

    pivot = work.pivot_table(index="name", columns="period", values="value", aggfunc="mean")
    pivot = pivot.reindex(columns=[p for p in period_order if p in pivot.columns])

    # Always sort by max-across-periods desc so the legend lists the most
    # prominent concepts first; then cap to max_concepts if requested.
    max_per_concept = pivot.max(axis=1, skipna=True)
    pivot = pivot.loc[max_per_concept.sort_values(ascending=False).index]
    if max_concepts is not None and max_concepts > 0:
        pivot = pivot.head(max_concepts)

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
                hovertemplate=(f"{name}<br>period: %{{x}}<br>{agg}: %{{y:.4f}}<extra></extra>"),
            )
        )

    suffix = (
        f" — top {max_concepts} concepts"
        if max_concepts is not None and max_concepts < len(pivot)
        else ""
    )
    fig.update_layout(
        title=title or f"Concept SHAP drift across periods ({agg}){suffix}",
        xaxis={
            "title": "period",
            "type": "category",
            "categoryorder": "array",
            "categoryarray": list(pivot.columns),
        },
        yaxis={"title": agg},
        legend={"title": "concept"},
        margin={"t": 60, "l": 70, "r": 30, "b": 60},
        height=max(360, 30 * len(pivot) + 200),
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
