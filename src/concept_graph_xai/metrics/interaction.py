"""Concept-level SHAP interaction matrix (P3).

Aggregates a feature × feature SHAP-interaction tensor into a concept × concept
matrix: each cell ``(Ci, Cj)`` is the per-sample sum of all interaction terms
between Ci's descendant features and Cj's descendants, aggregated across the
held-out set as ``mean(|.|)`` (absolute, default) or signed mean.

Diagonal cells ``(Ci, Ci)`` carry within-concept self-interaction (including
the main effects on the matrix's main diagonal).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import align_features

InteractionAgg = Literal["mean_abs", "mean_signed"]


def concept_interaction_matrix(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    shap_interaction_values: np.ndarray,
    *,
    only_concepts: bool = True,
    include_root: bool = False,
    agg: InteractionAgg = "mean_abs",
    on_unknown: str = "warn",
) -> pd.DataFrame:
    """Aggregate a per-sample feature × feature interaction tensor into concept × concept.

    Parameters
    ----------
    graph:
        ConceptGraph.
    feature_names:
        Names matching the second/third dimensions of ``shap_interaction_values``.
    shap_interaction_values:
        Tensor of shape ``(N, F, F)`` from
        ``shap.TreeExplainer(model).shap_interaction_values(X)``.
    only_concepts:
        If ``True`` (default), drop feature leaves from the matrix axes.
    include_root:
        If ``False`` (default), drop the root concept (it would otherwise
        sum every feature on both axes — uninformative).
    agg:
        ``"mean_abs"`` (default) reports ``mean_n |sum_{i,j} interaction|``
        — interaction *strength*. ``"mean_signed"`` reports
        ``mean_n sum_{i,j} interaction`` — net direction (can cancel).
    on_unknown:
        Behaviour when ``feature_names`` contains entries not present in the
        graph: ``"warn"`` (default), ``"ignore"``, ``"raise"``.

    Returns
    -------
    pandas.DataFrame
        Square DataFrame, rows and columns are concept names. ``df.attrs``
        carries ``agg`` and ``feature_count`` (per concept, dict).
    """

    arr = np.asarray(shap_interaction_values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(
            f"shap_interaction_values must be 3D (N, F, F); got shape {arr.shape}"
        )
    if arr.shape[1] != arr.shape[2]:
        raise ValueError(
            f"shap_interaction_values must have square last two dims; got {arr.shape[1:]}"
        )
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_interaction_values has {arr.shape[1]} features but feature_names has "
            f"{len(feature_names)}"
        )

    matched, indices, _missing = align_features(graph, feature_names, on_unknown=on_unknown)
    name_to_idx = {name: idx for name, idx in zip(matched, indices, strict=True)}

    nodes: list[str] = []
    for node in graph.nodes_in_order():
        if node == graph.root and not include_root:
            continue
        if only_concepts and graph.kind(node) == "feature":
            continue
        feats = [f for f in graph.descendant_features(node) if f in name_to_idx]
        if not feats:
            continue
        nodes.append(node)

    if not nodes:
        raise ValueError("no concepts (or features) overlap the supplied feature_names")

    feat_idxs: dict[str, list[int]] = {
        node: [name_to_idx[f] for f in graph.descendant_features(node) if f in name_to_idx]
        for node in nodes
    }
    feat_counts: dict[str, int] = {node: len(feat_idxs[node]) for node in nodes}

    n_concepts = len(nodes)
    matrix = np.zeros((n_concepts, n_concepts), dtype=float)
    for i, node_i in enumerate(nodes):
        idxs_i = feat_idxs[node_i]
        sub_i = arr[:, idxs_i, :]  # (N, |Ci|, F)
        for j, node_j in enumerate(nodes):
            if j < i:
                continue
            idxs_j = feat_idxs[node_j]
            cells = sub_i[:, :, idxs_j]  # (N, |Ci|, |Cj|)
            per_sample = cells.sum(axis=(1, 2))  # (N,)
            if agg == "mean_abs":
                value = float(np.abs(per_sample).mean())
            elif agg == "mean_signed":
                value = float(per_sample.mean())
            else:
                raise ValueError(f"unknown agg {agg!r}; expected 'mean_abs' or 'mean_signed'")
            matrix[i, j] = value
            matrix[j, i] = value

    df = pd.DataFrame(matrix, index=nodes, columns=nodes)
    df.attrs["agg"] = agg
    df.attrs["feature_count"] = feat_counts
    return df
