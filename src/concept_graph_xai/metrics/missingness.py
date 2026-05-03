"""Per-concept missingness metrics (P13, P15b).

* ``column_missing_rate`` — per-feature ``mean(X[col].isna())``.
* ``joint_missing_rate`` — per-concept ``mean(all features under concept are NaN
  in row)``. The headline number for the §H story: how often does a whole
  branch go missing at once?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import empty_concept_frame


def column_missing_rate(graph: ConceptGraph, X: pd.DataFrame) -> pd.DataFrame:
    """Mean per-feature missing rate, plus a concept-level *any-missing* mean.

    For a concept node, ``any_missing_rate`` is ``mean(any feature under the
    concept is NaN in row)`` — much weaker than :func:`joint_missing_rate`,
    but useful as an upper bound.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("column_missing_rate requires a pandas DataFrame X")
    df = empty_concept_frame(graph)
    column_rate: list[float] = []
    any_rate: list[float] = []
    n_rows = len(X)
    for node in graph.nodes_in_order():
        feats = [f for f in graph.descendant_features(node) if f in X.columns]
        if not feats:
            column_rate.append(0.0)
            any_rate.append(0.0)
            continue
        sub = X.loc[:, feats].isna()
        if graph.kind(node) == "feature":
            column_rate.append(float(sub.iloc[:, 0].mean()) if n_rows else 0.0)
        else:
            column_rate.append(float(sub.mean().mean()) if n_rows else 0.0)
        any_rate.append(float(sub.any(axis=1).mean()) if n_rows else 0.0)
    df["column_missing_rate"] = column_rate
    df["any_missing_rate"] = any_rate
    df["feature_count"] = [len(graph.descendant_features(n)) for n in graph.nodes_in_order()]
    return df


def joint_missing_rate(graph: ConceptGraph, X: pd.DataFrame) -> pd.DataFrame:
    """Per-concept joint-missing rate (P13/P15b).

    For each concept (and each feature, trivially), the fraction of rows where
    *every* feature under that concept is NaN. This is the number that should
    drive the "is the AUC-drop scenario realistic?" judgement.

    Returned columns: ``name``, ``kind``, ``depth``, ``parent``, ``feature_count``,
    ``joint_missing_rate``.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("joint_missing_rate requires a pandas DataFrame X")
    df = empty_concept_frame(graph)
    rate: list[float] = []
    feature_count: list[int] = []
    n_rows = len(X)
    for node in graph.nodes_in_order():
        feats = [f for f in graph.descendant_features(node) if f in X.columns]
        feature_count.append(len(feats))
        if not feats or n_rows == 0:
            rate.append(0.0)
            continue
        all_missing = X.loc[:, feats].isna().all(axis=1)
        rate.append(float(np.asarray(all_missing).mean()))
    df["feature_count"] = feature_count
    df["joint_missing_rate"] = rate
    return df
