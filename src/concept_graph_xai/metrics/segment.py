"""Per-segment per-concept SHAP aggregation (P7)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import (
    aligned_index_map,
    empty_concept_frame,
    grouping_order,
    per_sample_per_concept,
    resolve_grouping,
)

SegmentAgg = Literal["mean_abs", "mean_signed"]


# Back-compat aliases for v0.6 importers (e.g. concept_disparity used to
# pull these from metrics.segment). Will be removed once we're sure no
# external code imports them.
_resolve_segments = resolve_grouping
_segment_order = grouping_order


def segment_importance(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    segments: pd.Series | str,
    *,
    X: pd.DataFrame | None = None,
    agg: SegmentAgg = "mean_abs",
    on_unknown: str = "warn",
) -> pd.DataFrame:
    """Per-segment per-concept SHAP aggregate (long-form).

    For each ``(concept, segment)`` pair, computes ``mean_n agg(s_c[n])``
    over the rows belonging to ``segment``, where ``s_c[n] = sum_{f in
    concept.descendants} SHAP[n, f]``.

    Parameters
    ----------
    graph:
        The ConceptGraph.
    feature_names:
        Names matching the columns of ``shap_values``.
    shap_values:
        Per-sample SHAP, shape ``(N, F)``.
    segments:
        Either a ``pd.Series`` (aligned to ``shap_values`` rows, NaNs ignored)
        or a column-name string referencing ``X``.
    X:
        DataFrame whose row order matches ``shap_values``. Required when
        ``segments`` is a string.
    agg:
        ``"mean_abs"`` (default) reports ``mean_n |s_c[n]|`` — magnitude
        per segment. ``"mean_signed"`` reports ``mean_n s_c[n]`` — net
        signed direction (cancellation can shrink it).
    on_unknown:
        Behaviour when ``feature_names`` contains entries not in the graph.

    Returns
    -------
    pandas.DataFrame
        Long-form with columns ``name``, ``kind``, ``depth``, ``parent``,
        ``segment``, ``value``, ``feature_count``. ``df.attrs`` carries the
        ``agg`` choice and the stable ``segment_order`` list.
    """

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"shap_values must be 2D (N, F); got {arr.shape}")
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {arr.shape[1]} cols but feature_names has {len(feature_names)}"
        )

    seg_series = resolve_grouping(segments, X, arr.shape[0], param_name="segments")
    segment_order = grouping_order(seg_series)
    if not segment_order:
        raise ValueError("segments must contain at least one non-NA category")

    name_to_idx = aligned_index_map(graph, feature_names, on_unknown=on_unknown)

    nodes = graph.nodes_in_order()
    base = empty_concept_frame(graph)
    per_sample_per_node, feature_counts = per_sample_per_concept(graph, arr, name_to_idx)

    notna_mask = seg_series.notna().to_numpy()
    seg_strings = seg_series.astype(str)

    rows: list[dict[str, object]] = []
    for segment in segment_order:
        mask = notna_mask & (seg_strings.to_numpy() == segment)
        if not mask.any():
            continue
        block = per_sample_per_node[mask]
        if agg == "mean_abs":
            values = np.abs(block).mean(axis=0)
        elif agg == "mean_signed":
            values = block.mean(axis=0)
        else:
            raise ValueError(f"unknown agg {agg!r}; expected 'mean_abs' or 'mean_signed'")
        for k in range(len(nodes)):
            rows.append(
                {
                    "name": base.iloc[k]["name"],
                    "kind": base.iloc[k]["kind"],
                    "depth": base.iloc[k]["depth"],
                    "parent": base.iloc[k]["parent"],
                    "path": base.index[k],
                    "segment": segment,
                    "value": float(values[k]),
                    "feature_count": int(feature_counts[k]),
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["agg"] = agg
    df.attrs["segment_order"] = segment_order
    return df
