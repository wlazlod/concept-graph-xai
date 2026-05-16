"""Sunburst variant where each concept is coloured by a metadata tag (P12)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import (
    build_sunburst_figure,
    hover_text,
    sunburst_layout,
)

_DEFAULT_PALETTE: list[str] = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def regulatory_tag_overlay(
    graph: ConceptGraph,
    df: pd.DataFrame | None = None,
    *,
    tag_key: str = "tag",
    palette: dict[str, str] | None = None,
    untagged_color: str = "#dddddd",
    value: str = "count",
    hide_root: bool = True,
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a sunburst whose sectors are coloured by a node-metadata tag.

    Parameters
    ----------
    graph:
        ConceptGraph; tag is read from ``graph.view(node).metadata[tag_key]``.
    df:
        Optional DataFrame providing the ``feature_count`` column (or any
        ``value`` column). Defaults to a count-based sunburst.
    tag_key:
        Metadata key carrying the categorical tag.
    palette:
        Optional ``tag -> css_color`` mapping. Unmapped tags get colours from
        a default palette.
    untagged_color:
        Colour for nodes that carry no value under ``tag_key``.
    hide_root:
        When ``True`` (default) the root concept is omitted; pass ``False``
        to keep the legacy root sector.
    """

    if df is None:
        from concept_graph_xai.metrics.counts import feature_counts

        df = feature_counts(graph)

    arrays, ordered, sizes = sunburst_layout(graph, df, value=value, hide_root=hide_root)

    rendered_nodes = [
        node for node in graph.nodes_in_order() if not (hide_root and node == graph.root)
    ]
    tags: list[str] = []
    for node in rendered_nodes:
        meta = graph.view(node).metadata
        tag = meta.get(tag_key)
        tags.append(str(tag) if tag is not None else "")

    palette_map = dict(palette) if palette else {}
    next_idx = 0
    for tag in tags:
        if tag and tag not in palette_map:
            palette_map[tag] = _DEFAULT_PALETTE[next_idx % len(_DEFAULT_PALETTE)]
            next_idx += 1

    colors = [palette_map.get(tag, untagged_color) for tag in tags]
    hover = hover_text(ordered.assign(tag=tags), [value, "tag"])

    return build_sunburst_figure(
        arrays,
        sizes,
        marker={"colors": colors},
        hover=hover,
        title=title or f"Concepts coloured by {tag_key!r}",
        layout_kwargs=layout_kwargs,
    )
