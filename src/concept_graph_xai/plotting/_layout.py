"""Internal helpers that build Plotly-friendly arrays from a ConceptGraph."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from concept_graph_xai.graph import ConceptGraph


def graph_to_arrays(graph: ConceptGraph) -> dict[str, list[str]]:
    """Build the ``ids``, ``labels``, ``parents`` arrays Plotly sunburst needs.

    Uses ``"/".join(path)`` as a stable id so that label collisions across
    branches are impossible. ``labels`` are the bare node names. ``parents``
    are the parent ids (``""`` for the root).
    """

    ids: list[str] = []
    labels: list[str] = []
    parents: list[str] = []
    for node in graph.nodes_in_order():
        path = graph.path(node)
        ids.append("/".join(path))
        labels.append(node)
        parents.append("/".join(path[:-1]))
    return {"ids": ids, "labels": labels, "parents": parents}


def reindex_to_paths(df: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
    """Reorder a metrics DataFrame to match a given list of ids (concept paths)."""

    if df.index.name != "path":
        raise ValueError("expected DataFrame indexed by 'path' (run a metric function first)")
    return df.reindex(list(ids))


def hover_text(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    fmt: dict[str, str] | None = None,
) -> list[str]:
    """Build a ``customdata`` and matching ``hovertemplate``-friendly string list.

    Returns one string per row, joining the requested columns as
    ``"<col>: <formatted value>"``.
    """

    fmt = fmt or {}
    out: list[str] = []
    for _, row in df.iterrows():
        parts: list[str] = []
        for col in columns:
            if col not in row.index:
                continue
            value = row[col]
            if pd.isna(value):
                continue
            spec = fmt.get(col)
            text = format(value, spec) if spec else str(value)
            parts.append(f"{col}: {text}")
        out.append("<br>".join(parts))
    return out
