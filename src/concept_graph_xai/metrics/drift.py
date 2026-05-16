"""Per-period concept SHAP aggregation for drift monitoring (P9, P10).

Two related metrics:

* :func:`attribution_drift` — long-form table of per-period per-concept
  aggregated SHAP. Consumed by :func:`concept_drift_lines`.
* :func:`concept_drift_delta` — wide table of two periods plus their delta.
  Consumed by :func:`concept_drift_sunburst`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import align_features, empty_concept_frame

DriftAgg = Literal["mean_abs", "mean_signed"]
PeriodSpec = tuple[str, np.ndarray, Sequence[str]]


def _aggregate_period(
    graph: ConceptGraph,
    period_label: str,
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    agg: DriftAgg,
    on_unknown: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(per_node_values, per_node_feature_counts)`` for one period."""

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(
            f"shap_values for period {period_label!r} must be 2D (N, F); got {arr.shape}"
        )
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values for period {period_label!r} has {arr.shape[1]} cols "
            f"but feature_names has {len(feature_names)}"
        )
    matched, indices, _missing = align_features(graph, feature_names, on_unknown=on_unknown)
    name_to_idx = {name: idx for name, idx in zip(matched, indices, strict=True)}

    nodes = graph.nodes_in_order()
    values = np.zeros(len(nodes), dtype=float)
    feature_counts = np.zeros(len(nodes), dtype=int)
    for k, node in enumerate(nodes):
        feats = [f for f in graph.descendant_features(node) if f in name_to_idx]
        feature_counts[k] = len(feats)
        if not feats:
            continue
        cols = [name_to_idx[f] for f in feats]
        s = arr[:, cols].sum(axis=1)
        if agg == "mean_abs":
            values[k] = float(np.abs(s).mean())
        elif agg == "mean_signed":
            values[k] = float(s.mean())
        else:
            raise ValueError(f"unknown agg {agg!r}; expected 'mean_abs' or 'mean_signed'")
    return values, feature_counts


def attribution_drift(
    graph: ConceptGraph,
    periods: Sequence[PeriodSpec],
    *,
    agg: DriftAgg = "mean_abs",
    on_unknown: str = "warn",
) -> pd.DataFrame:
    """Per-period per-concept SHAP aggregate (long-form).

    Parameters
    ----------
    graph:
        The ConceptGraph.
    periods:
        Ordered list of ``(period_label, shap_values, feature_names)`` tuples.
        Sample counts may differ across periods; feature counts must match
        ``feature_names`` for each entry.
    agg:
        ``"mean_abs"`` (default) reports ``mean_n |s_c[n]|`` — magnitude
        per period. ``"mean_signed"`` reports ``mean_n s_c[n]`` — net
        signed direction.
    on_unknown:
        Behaviour when ``feature_names`` contains entries not in the graph.

    Returns
    -------
    pandas.DataFrame
        Long-form with columns ``name``, ``kind``, ``depth``, ``parent``,
        ``path``, ``period``, ``value``, ``feature_count``. ``df.attrs``
        carries the ``agg`` choice and the input ``period_order`` list.
    """

    if not periods:
        raise ValueError("periods must contain at least one entry")
    period_labels = [str(p[0]) for p in periods]
    if len(set(period_labels)) != len(period_labels):
        raise ValueError(f"period labels must be unique; got {period_labels}")

    base = empty_concept_frame(graph)
    rows: list[dict[str, object]] = []
    for period_label, shap_values, feature_names in periods:
        values, feature_counts = _aggregate_period(
            graph, str(period_label), shap_values, feature_names, agg, on_unknown
        )
        for k in range(len(base)):
            rows.append(
                {
                    "name": base.iloc[k]["name"],
                    "kind": base.iloc[k]["kind"],
                    "depth": base.iloc[k]["depth"],
                    "parent": base.iloc[k]["parent"],
                    "path": base.index[k],
                    "period": str(period_label),
                    "value": float(values[k]),
                    "feature_count": int(feature_counts[k]),
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["agg"] = agg
    df.attrs["period_order"] = period_labels
    return df
