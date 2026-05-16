"""Multi-tier SHAP flow Sankey: features → ... → top-level concept → ±outcome (P4)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.plotting._layout import branch_colors, hex_to_rgb


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = hex_to_rgb(hex_color)
    return f"rgba({round(r * 255)},{round(g * 255)},{round(b * 255)},{alpha:.2f})"


def concept_sankey(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    *,
    max_features_per_concept: int | None = None,
    branch_palette: Sequence[str] | None = None,
    positive_color: str = "#2ca02c",
    negative_color: str = "#d62728",
    title: str | None = None,
    layout_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Render a SHAP-flow Sankey that walks the **full** concept hierarchy.

    Tiers (left → right):

    * **Features** — one node per included feature.
    * **Concept tiers** — one node per concept on any included feature's
      path (excluding the hidden root). For deep trees this produces
      multiple intermediate tiers (e.g. ``feature → sub-concept →
      top-level concept``).
    * **Outcome** — two nodes (+, −).

    Link weights:

    * ``feature → direct-parent``: ``sum_n |SHAP[n, f]|`` — total
      magnitude carried by the feature.
    * ``concept → its parent concept``: ``sum_n |s_c[n]|`` where
      ``s_c[n] = sum_{f in c.descendants} SHAP[n, f]``. This is the
      *signed* sum's magnitude, so within-concept cancellation
      narrows the band as you move right.
    * ``top-level concept → +``: ``sum_n max(0, s_t[n])``.
    * ``top-level concept → −``: ``sum_n max(0, -s_t[n])``.

    Conservation is intentionally relaxed: at each concept node, total
    *incoming* flow ≥ *outgoing* flow whenever descendants' SHAP
    cancel within the concept. The shrinkage from left to right is the
    diagnostic — it tells you which concepts wash out under aggregation.

    Parameters
    ----------
    graph:
        The ConceptGraph.
    feature_names:
        Names matching the columns of ``shap_values``.
    shap_values:
        Per-sample SHAP, shape ``(N, F)``.
    max_features_per_concept:
        Optional cap: per top-level concept, keep only the top-K
        descendant features by ``sum_n |SHAP|``. Concepts on the
        ancestor chain of kept features remain.
    branch_palette:
        Custom palette for branch base hues.
    positive_color, negative_color:
        Colours for the ``+`` / ``−`` outcome nodes (and their incoming
        links).
    title:
        Figure title.
    layout_kwargs:
        Passed verbatim to ``fig.update_layout``.
    """

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"shap_values must be 2D (N, F); got {arr.shape}")
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {arr.shape[1]} cols but feature_names has {len(feature_names)}"
        )

    name_to_col: dict[str, int] = {name: i for i, name in enumerate(feature_names)}

    # 1. Build per-feature ancestor chain (feature → direct parent → ... → top-level).
    #    Drop features with no concept ancestor (path is just (root, feature)) and
    #    features carrying zero SHAP magnitude.
    feature_magnitude: dict[str, float] = {}
    feature_chain: dict[str, list[str]] = {}
    feature_top_branch: dict[str, str] = {}
    for feat in graph.features():
        col = name_to_col.get(feat)
        if col is None:
            continue
        path = graph.path(feat)
        if len(path) < 3:
            continue  # feature directly under root → no concept layer to render
        magnitude = float(np.abs(arr[:, col]).sum())
        if magnitude == 0.0:
            continue
        feature_magnitude[feat] = magnitude
        # chain: [feature, direct_parent, ..., top-level concept]
        chain = [path[-1], *list(reversed(path[1:-1]))]
        feature_chain[feat] = chain
        feature_top_branch[feat] = path[1]

    if not feature_magnitude:
        raise ValueError(
            "no features carry non-zero SHAP magnitude under a concept; "
            "check feature_names alignment and graph structure"
        )

    # 2. Optional: cap features per top-level concept by magnitude.
    if max_features_per_concept is not None:
        per_branch: dict[str, list[str]] = {}
        for feat, branch in feature_top_branch.items():
            per_branch.setdefault(branch, []).append(feat)
        kept: set[str] = set()
        for feats in per_branch.values():
            feats_sorted = sorted(feats, key=lambda f: feature_magnitude[f], reverse=True)
            kept.update(feats_sorted[:max_features_per_concept])
        feature_magnitude = {f: m for f, m in feature_magnitude.items() if f in kept}
        feature_chain = {f: c for f, c in feature_chain.items() if f in kept}
        feature_top_branch = {f: b for f, b in feature_top_branch.items() if f in kept}
        if not feature_magnitude:
            raise ValueError("max_features_per_concept filtered out every feature")

    # 3. Concept set: every node appearing on any chain (excluding the feature).
    included_concepts_in_order: list[str] = []
    seen_concepts: set[str] = set()
    for chain in feature_chain.values():
        for node in chain[1:]:
            if node not in seen_concepts:
                seen_concepts.add(node)
                included_concepts_in_order.append(node)

    # 4. Per-concept signed per-sample value (sum of SHAP across its included
    #    descendant features). Used for both concept→parent magnitudes and
    #    top-level→outcome signed splits.
    concept_signed_per_sample: dict[str, np.ndarray] = {}
    for concept in seen_concepts:
        feats = [
            f
            for f in graph.descendant_features(concept)
            if f in feature_magnitude
        ]
        if not feats:
            continue
        cols = [name_to_col[f] for f in feats]
        concept_signed_per_sample[concept] = arr[:, cols].sum(axis=1)

    # 5. Build node list, color map, and explicit (x, y) positions.
    #    Plotly's automatic layout minimizes link crossings and re-orders
    #    nodes within a tier — so we pin every node with explicit (x, y)
    #    instead. Vertical order within every tier follows the graph's DFS
    #    preorder, which groups every node next to its siblings under the
    #    same parent.
    node_labels: list[str] = []
    node_colors: list[str] = []
    feature_node_idx: dict[str, int] = {}
    concept_node_idx: dict[str, int] = {}

    branch_set = {feature_top_branch[f] for f in feature_magnitude}
    branch_ids = ["/".join(graph.path(b)) for b in branch_set]
    branch_colors_list = branch_colors(graph, branch_ids, palette=branch_palette)
    color_for_branch_path = dict(zip(branch_ids, branch_colors_list, strict=True))

    def node_color(top_branch: str) -> str:
        return color_for_branch_path.get("/".join(graph.path(top_branch)), "#cccccc")

    dfs_order = graph.nodes_in_order()
    feature_set = set(feature_magnitude)
    max_concept_depth = max(
        (len(graph.path(c)) - 1 for c in seen_concepts), default=1
    )
    # Tier x-positions in (0, 1), strictly increasing left -> right:
    # tier 0 = features, tiers 1..max_concept_depth = concepts (deepest -> top),
    # final tier = outcomes.
    n_tiers = 1 + max_concept_depth + 1  # features + concept tiers + outcome
    margin = 1.0 / (n_tiers + 1)
    tier_x: dict[int, float] = {
        t: margin + t * (1.0 - 2 * margin) / max(n_tiers - 1, 1) for t in range(n_tiers)
    }

    # 5a. Features (tier 0) in DFS preorder -> ontological grouping
    feature_tier: list[str] = [n for n in dfs_order if n in feature_set]
    # 5b. Concepts grouped by depth, deepest first; within each depth, DFS order
    concept_tiers: dict[int, list[str]] = {}
    for current_depth in range(max_concept_depth, 0, -1):
        concept_tiers[current_depth] = [
            n
            for n in dfs_order
            if n in seen_concepts and len(graph.path(n)) - 1 == current_depth
        ]

    # Map depth -> tier index. tier 0 = features; tier t (1..max_concept_depth)
    # = concepts of depth (max_concept_depth - t + 1) so deepest concepts are
    # tier 1 (just right of features) and top-level concepts are tier
    # max_concept_depth (just left of outcomes).
    def concept_tier_index(depth: int) -> int:
        return max_concept_depth - depth + 1

    def y_positions(n: int) -> list[float]:
        if n == 1:
            return [0.5]
        # Plotly Sankey uses y=0 at the top and y=1 at the bottom (screen
        # coordinates, not Cartesian) — so rank 0 (first in DFS) gets the
        # smallest y and sits at the top of the diagram.
        return [(i + 0.5) / n for i in range(n)]

    node_x: list[float] = []
    node_y: list[float] = []

    for feat, y in zip(feature_tier, y_positions(len(feature_tier)), strict=True):
        feature_node_idx[feat] = len(node_labels)
        node_labels.append(feat)
        node_colors.append(node_color(feature_top_branch[feat]))
        node_x.append(tier_x[0])
        node_y.append(y)

    for depth in range(max_concept_depth, 0, -1):
        tier_nodes = concept_tiers[depth]
        tier_idx = concept_tier_index(depth)
        ys = y_positions(len(tier_nodes))
        for node, y in zip(tier_nodes, ys, strict=True):
            concept_node_idx[node] = len(node_labels)
            node_labels.append(node)
            node_colors.append(node_color(graph.path(node)[1]))
            node_x.append(tier_x[tier_idx])
            node_y.append(y)

    # + outcome on top, - outcome at the bottom (y=0 is the top in Plotly).
    pos_node_idx = len(node_labels)
    node_labels.append("+ outcome")
    node_colors.append(positive_color)
    node_x.append(tier_x[n_tiers - 1])
    node_y.append(0.25)
    neg_node_idx = len(node_labels)
    node_labels.append("- outcome")
    node_colors.append(negative_color)
    node_x.append(tier_x[n_tiers - 1])
    node_y.append(0.75)

    # 6. Build links.
    sources: list[int] = []
    targets: list[int] = []
    values: list[float] = []
    link_colors: list[str] = []

    # 6a. feature → direct parent
    for feat in feature_magnitude:
        chain = feature_chain[feat]
        direct_parent = chain[1]
        sources.append(feature_node_idx[feat])
        targets.append(concept_node_idx[direct_parent])
        values.append(feature_magnitude[feat])
        link_colors.append(_rgba(node_color(feature_top_branch[feat]), 0.35))

    # 6b. concept → its parent concept (skip top-level: their parent is the root)
    for concept in seen_concepts:
        path = graph.path(concept)
        if len(path) <= 2:
            continue  # top-level concept (its parent is root, handled at outcome step)
        parent = path[-2]
        if parent not in concept_node_idx:
            continue
        per_sample = concept_signed_per_sample.get(concept)
        if per_sample is None:
            continue
        weight = float(np.abs(per_sample).sum())
        if weight == 0.0:
            continue
        sources.append(concept_node_idx[concept])
        targets.append(concept_node_idx[parent])
        values.append(weight)
        link_colors.append(_rgba(node_color(path[1]), 0.35))

    # 6c. top-level concept → +/- outcome
    top_levels = {feature_top_branch[f] for f in feature_magnitude}
    for top in top_levels:
        per_sample = concept_signed_per_sample.get(top)
        if per_sample is None:
            continue
        pos_flow = float(np.maximum(per_sample, 0.0).sum())
        neg_flow = float(np.maximum(-per_sample, 0.0).sum())
        if pos_flow > 0:
            sources.append(concept_node_idx[top])
            targets.append(pos_node_idx)
            values.append(pos_flow)
            link_colors.append(_rgba(positive_color, 0.35))
        if neg_flow > 0:
            sources.append(concept_node_idx[top])
            targets.append(neg_node_idx)
            values.append(neg_flow)
            link_colors.append(_rgba(negative_color, 0.35))

    fig = go.Figure(
        go.Sankey(
            # arrangement="snap" respects explicit node x/y while still
            # snapping minor adjustments; "fixed" would freeze user dragging
            # too aggressively.
            arrangement="snap",
            node={
                "label": node_labels,
                "color": node_colors,
                "x": node_x,
                "y": node_y,
                "pad": 14,
                "thickness": 16,
                "line": {"color": "rgba(0,0,0,0.5)", "width": 0.5},
            },
            link={
                "source": sources,
                "target": targets,
                "value": values,
                "color": link_colors,
            },
        )
    )

    fig.update_layout(
        title=title or "Concept SHAP flow — feature -> concepts -> +/- outcome",
        margin={"t": 60, "l": 30, "r": 30, "b": 30},
        height=max(420, 22 * len(node_labels) + 200),
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig
