"""Importance-aggregation metric: sum per-feature importance up the tree."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import (
    aggregate_per_feature,
    align_features,
    empty_concept_frame,
)


def importance_sum(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    importances: np.ndarray,
    *,
    signed: bool = False,
    agg: str = "mean",
    on_unknown: str = "warn",
) -> pd.DataFrame:
    """Sum per-feature importance under each concept.

    Parameters
    ----------
    graph:
        ConceptGraph whose features are a subset of ``feature_names``.
    feature_names:
        Names matching the columns of ``importances``.
    importances:
        ``(F,)`` per-feature aggregate or ``(N, F)`` per-sample SHAP-style array.
    signed:
        Pass-through to :func:`aggregate_per_feature`.
    agg:
        Pass-through to :func:`aggregate_per_feature`.
    on_unknown:
        Behaviour when input features are missing from the graph: ``"warn"``,
        ``"ignore"``, or ``"raise"``.
    """

    per_feature = aggregate_per_feature(importances, signed=signed, agg=agg)
    if per_feature.shape[0] != len(feature_names):
        raise ValueError(
            f"importance length {per_feature.shape[0]} != len(feature_names) {len(feature_names)}"
        )
    matched, indices, _missing = align_features(graph, feature_names, on_unknown=on_unknown)
    name_to_value = {
        name: float(per_feature[idx]) for name, idx in zip(matched, indices, strict=True)
    }

    df = empty_concept_frame(graph)
    sums: list[float] = []
    for node in graph.nodes_in_order():
        feats = graph.descendant_features(node)
        sums.append(float(sum(name_to_value.get(f, 0.0) for f in feats)))
    df["importance_sum"] = sums
    df["feature_count"] = [len(graph.descendant_features(n)) for n in graph.nodes_in_order()]
    return df
