"""Concept-coherence vs concept-importance scatter (P16)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_QUADRANT_COLOR: dict[str, str] = {
    "well_designed": "#2ca02c",
    "kitchen_sink": "#d62728",
    "redundant": "#ff7f0e",
    "noise": "#7f7f7f",
    "undefined": "#cccccc",
}


def coherence_importance_scatter(
    df: pd.DataFrame,
    *,
    only_concepts: bool = True,
    label_points: bool = True,
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render the coherence × importance quadrant scatter.

    Parameters
    ----------
    df:
        Output of :func:`coherence_importance`. Must carry ``coherence``,
        ``importance_sum`` and ``quadrant`` columns. Threshold values are
        read from ``df.attrs["coherence_threshold"]`` and
        ``df.attrs["importance_threshold"]``.
    only_concepts:
        Drop rows where ``kind == "feature"`` so the chart shows only
        business concepts.
    label_points:
        Annotate every point with the concept name.
    """

    needed = {"coherence", "importance_sum", "quadrant"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"missing columns from coherence_importance: {missing}")

    plot_df = df.copy()
    if only_concepts and "kind" in plot_df.columns:
        plot_df = plot_df.loc[plot_df["kind"] == "concept"].copy()

    coh_thr = float(plot_df.attrs.get("coherence_threshold", df.attrs.get("coherence_threshold", 0.0)))
    imp_thr = float(plot_df.attrs.get("importance_threshold", df.attrs.get("importance_threshold", 0.0)))

    fig = go.Figure()
    for quadrant, color in _QUADRANT_COLOR.items():
        sub = plot_df.loc[plot_df["quadrant"] == quadrant]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["coherence"],
                y=sub["importance_sum"],
                mode="markers+text" if label_points else "markers",
                text=sub["name"] if label_points else None,
                textposition="top center",
                marker={
                    "size": 12,
                    "color": color,
                    "line": {"color": "black", "width": 0.5},
                },
                name=quadrant.replace("_", " "),
                customdata=np.stack(
                    [
                        sub.get("feature_count", pd.Series([0] * len(sub))).to_numpy(),
                        sub.get("kind", pd.Series([""] * len(sub))).to_numpy(),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "coherence: %{x:.3f}<br>"
                    "importance: %{y:.4f}<br>"
                    "feature_count: %{customdata[0]}<br>"
                    "kind: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.add_hline(y=imp_thr, line={"color": "black", "dash": "dash", "width": 1})
    fig.add_vline(x=coh_thr, line={"color": "black", "dash": "dash", "width": 1})

    fig.update_layout(
        title=title or "Concept coherence vs importance",
        xaxis_title=f"within-concept mean(|ρ|)  ({df.attrs.get('method', 'spearman')})",
        yaxis_title="summed |SHAP|",
        margin={"t": 60, "l": 60, "r": 30, "b": 60},
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
