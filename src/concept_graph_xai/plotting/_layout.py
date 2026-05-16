"""Internal helpers that build Plotly-friendly arrays from a ConceptGraph."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from concept_graph_xai.graph import ConceptGraph

# Plotly qualitative.Plotly palette, inlined so we don't pay a plotly.colors
# import at metric-import time. Used as the default branch palette.
_DEFAULT_BRANCH_PALETTE: tuple[str, ...] = (
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

_ROOT_FALLBACK_COLOR: str = "#cccccc"


def graph_to_arrays(
    graph: ConceptGraph,
    *,
    hide_root: bool = False,
) -> dict[str, list[str]]:
    """Build the ``ids``, ``labels``, ``parents`` arrays Plotly sunburst needs.

    Uses ``"/".join(path)`` as a stable id so that label collisions across
    branches are impossible. ``labels`` are the bare node names. ``parents``
    are the parent ids (``""`` for the root).

    When ``hide_root=True``, the root node is omitted from the arrays and its
    direct children are re-parented to ``""``. Plotly renders multiple
    ``parent=""`` nodes as adjacent wedges of the full circle, so the visual
    effect is "the root sector disappears and its children become the centre
    ring".
    """

    ids: list[str] = []
    labels: list[str] = []
    parents: list[str] = []
    root_id = graph.root
    for node in graph.nodes_in_order():
        if hide_root and node == root_id:
            continue
        path = graph.path(node)
        ids.append("/".join(path))
        labels.append(node)
        parent_path = path[:-1]
        if hide_root and parent_path == (root_id,):
            parents.append("")
        else:
            parents.append("/".join(parent_path))
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


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def rgb_to_hex(r: float, g: float, b: float) -> str:
    ri = max(0, min(255, round(r * 255)))
    gi = max(0, min(255, round(g * 255)))
    bi = max(0, min(255, round(b * 255)))
    return f"#{ri:02x}{gi:02x}{bi:02x}"


def _lighten(hex_color: str, factor: float) -> str:
    """Linear blend of ``hex_color`` toward white by ``factor`` ∈ [0, 1]."""

    factor = max(0.0, min(1.0, factor))
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_hex(r + (1.0 - r) * factor, g + (1.0 - g) * factor, b + (1.0 - b) * factor)


def branch_colors(
    graph: ConceptGraph,
    ids: Sequence[str],
    *,
    palette: Sequence[str] | None = None,
    root_color: str = _ROOT_FALLBACK_COLOR,
    max_lighten: float = 0.55,
) -> list[str]:
    """Return one CSS color per id, with hierarchical shading per top-level branch.

    Each top-level concept (``path[1]``) gets a base hue from ``palette``. Its
    descendants inherit that hue and are progressively lightened with depth: the
    top-level branch itself is the saturated base, sub-concepts are lighter,
    leaf features are lightest. The maximum lightening is bounded by
    ``max_lighten`` (default ``0.55``, i.e. leaves are blended 55% toward white)
    so even the deepest level stays visible against a white background.

    Nodes that *are* the root (e.g. when the chart is rendered with
    ``hide_root=False``) get ``root_color``.
    """

    pal: tuple[str, ...] = tuple(palette) if palette else _DEFAULT_BRANCH_PALETTE
    if not pal:
        raise ValueError("palette must contain at least one color")

    branch_for_color: dict[str, str] = {}
    branch_max_depth: dict[str, int] = {}
    next_idx = 0
    for node in graph.nodes_in_order():
        path = graph.path(node)
        if len(path) < 2:
            continue
        branch = path[1]
        if branch not in branch_for_color:
            branch_for_color[branch] = pal[next_idx % len(pal)]
            next_idx += 1
        depth_within_branch = len(path) - 2  # branch itself = 0
        if depth_within_branch > branch_max_depth.get(branch, 0):
            branch_max_depth[branch] = depth_within_branch

    root_id = graph.root
    out: list[str] = []
    for sector_id in ids:
        parts = sector_id.split("/")
        if len(parts) < 2 or parts[0] != root_id:
            out.append(root_color)
            continue
        branch = parts[1]
        base = branch_for_color.get(branch)
        if base is None:
            out.append(root_color)
            continue
        depth_within_branch = len(parts) - 2
        max_d = branch_max_depth.get(branch, 0)
        factor = (depth_within_branch / max_d) * max_lighten if max_d > 0 else 0.0
        out.append(_lighten(base, factor))
    return out
