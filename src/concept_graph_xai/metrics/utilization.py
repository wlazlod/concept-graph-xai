"""Utilization metric: which parts of the graph the model actually uses."""

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


def utilization(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    importances: np.ndarray,
    *,
    threshold: float = 0.0,
    signed: bool = False,
    agg: str = "mean",
    on_unknown: str = "warn",
) -> pd.DataFrame:
    """Mark each node ``is_used`` when its aggregated importance exceeds ``threshold``.

    A *feature* node is used iff ``abs(importance) > threshold`` (when
    ``signed=False``). A *concept* node is used iff any of its feature
    descendants is used.

    Returned DataFrame columns: ``name``, ``kind``, ``depth``, ``parent``,
    ``importance_sum``, ``feature_count``, ``used_feature_count``, ``is_used``.
    """

    per_feature = aggregate_per_feature(importances, signed=signed, agg=agg)
    if per_feature.shape[0] != len(feature_names):
        raise ValueError(
            f"importance length {per_feature.shape[0]} != len(feature_names) {len(feature_names)}"
        )
    matched, indices, _missing = align_features(graph, feature_names, on_unknown=on_unknown)
    name_to_value = {name: float(per_feature[idx]) for name, idx in zip(matched, indices, strict=True)}

    used_features: set[str] = set()
    for name, value in name_to_value.items():
        magnitude = abs(value) if not signed else value
        if magnitude > threshold:
            used_features.add(name)

    df = empty_concept_frame(graph)
    feature_count: list[int] = []
    used_feature_count: list[int] = []
    importance_sum_col: list[float] = []
    is_used_col: list[bool] = []
    for node in graph.nodes_in_order():
        feats = graph.descendant_features(node)
        used = [f for f in feats if f in used_features]
        feature_count.append(len(feats))
        used_feature_count.append(len(used))
        importance_sum_col.append(float(sum(name_to_value.get(f, 0.0) for f in feats)))
        is_used_col.append(len(used) > 0)
    df["importance_sum"] = importance_sum_col
    df["feature_count"] = feature_count
    df["used_feature_count"] = used_feature_count
    df["is_used"] = is_used_col
    return df
