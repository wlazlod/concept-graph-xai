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
from concept_graph_xai.metrics._common import (
    aligned_index_map,
    empty_concept_frame,
    per_sample_per_concept,
)

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
    name_to_idx = aligned_index_map(graph, feature_names, on_unknown=on_unknown)

    per_sample, feature_counts = per_sample_per_concept(graph, arr, name_to_idx)
    if agg == "mean_abs":
        values = np.abs(per_sample).mean(axis=0)
    elif agg == "mean_signed":
        values = per_sample.mean(axis=0)
    else:
        raise ValueError(f"unknown agg {agg!r}; expected 'mean_abs' or 'mean_signed'")
    return values.astype(float), feature_counts


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


def concept_drift_delta(
    graph: ConceptGraph,
    periods: Sequence[PeriodSpec],
    *,
    baseline: str | None = None,
    target: str | None = None,
    agg: DriftAgg = "mean_abs",
    on_unknown: str = "warn",
) -> pd.DataFrame:
    """Per-concept baseline / target / delta between two periods.

    Convenience wrapper around :func:`attribution_drift` for the
    two-period delta view consumed by :func:`concept_drift_sunburst`.

    Parameters
    ----------
    graph:
        The ConceptGraph.
    periods:
        Same as :func:`attribution_drift` — list of
        ``(period_label, shap_values, feature_names)`` tuples.
    baseline:
        Period label to treat as the reference. Defaults to the first
        period.
    target:
        Period label to compare against. Defaults to the last period.
    agg:
        Aggregation passed through to :func:`attribution_drift`.
    on_unknown:
        Pass-through.

    Returns
    -------
    pandas.DataFrame
        Indexed by concept path, columns ``name``, ``kind``, ``depth``,
        ``parent``, ``baseline``, ``target``, ``delta``,
        ``feature_count``. ``df.attrs`` carries the ``agg``,
        ``baseline_period``, ``target_period`` strings.
    """

    if not periods:
        raise ValueError("periods must contain at least one entry")
    period_labels = [str(p[0]) for p in periods]
    baseline_label = period_labels[0] if baseline is None else str(baseline)
    target_label = period_labels[-1] if target is None else str(target)
    if baseline_label not in period_labels:
        raise KeyError(f"baseline period {baseline_label!r} not in periods: {period_labels}")
    if target_label not in period_labels:
        raise KeyError(f"target period {target_label!r} not in periods: {period_labels}")
    if baseline_label == target_label:
        raise ValueError(f"baseline and target must differ; both set to {baseline_label!r}")

    long_df = attribution_drift(graph, periods, agg=agg, on_unknown=on_unknown)
    base_df = long_df[long_df["period"] == baseline_label].set_index("path")
    targ_df = long_df[long_df["period"] == target_label].set_index("path")

    base = empty_concept_frame(graph)
    base["baseline"] = base_df["value"].reindex(base.index).to_numpy(dtype=float)
    base["target"] = targ_df["value"].reindex(base.index).to_numpy(dtype=float)
    base["delta"] = base["target"] - base["baseline"]
    base["feature_count"] = base_df["feature_count"].reindex(base.index).to_numpy(dtype=int)
    base.attrs["agg"] = agg
    base.attrs["baseline_period"] = baseline_label
    base.attrs["target_period"] = target_label
    return base
