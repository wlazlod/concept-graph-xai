"""Per-concept SHAP disparity vs a reference protected group (P11).

A direct cousin of :func:`segment_importance`: takes the same Series-or-
column-name protected-attribute vector and computes per-group per-concept
aggregates, then subtracts the reference group's row to yield the *gap*
(additive disparity). The reference row is preserved with ``value=0`` so
the heatmap can show it as a visible baseline.

The v0.6 API is intentionally minimal — a single protected attribute per
call. Intersections (e.g. ``gender × age_band``) and a first-class
``ProtectedAttribute`` object on the graph are deferred until a real
request lands; users who need either can pre-aggregate their attributes
into one ``pd.Series`` and call this twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import align_features, empty_concept_frame
from concept_graph_xai.metrics.segment import _resolve_segments, _segment_order

DisparityAgg = Literal["mean_abs", "mean_signed"]


def concept_disparity(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    protected: pd.Series | str,
    *,
    reference: str,
    X: pd.DataFrame | None = None,
    agg: DisparityAgg = "mean_abs",
    on_unknown: str = "warn",
) -> pd.DataFrame:
    """Per-concept additive SHAP gap vs a reference protected group.

    For each ``(concept, group)`` pair, computes
    ``mean_n agg(s_c[n]) over rows in group`` minus the same quantity for
    ``reference``. The reference group's row exists in the output with
    ``value=0`` for every concept.

    Parameters
    ----------
    graph:
        The ConceptGraph.
    feature_names:
        Names matching the columns of ``shap_values``.
    shap_values:
        Per-sample SHAP, shape ``(N, F)``.
    protected:
        Either a ``pd.Series`` aligned to ``shap_values`` rows (NaNs are
        ignored) or a column-name string referencing ``X``.
    reference:
        Label of the baseline group. Must appear in the resolved
        protected-attribute values.
    X:
        DataFrame whose row order matches ``shap_values``. Required when
        ``protected`` is a string.
    agg:
        ``"mean_abs"`` (default) measures whether the model *relies* on
        the concept differently across groups — magnitude differential.
        ``"mean_signed"`` measures whether the concept *pushes
        predictions* in different directions across groups — treatment
        differential.
    on_unknown:
        Behaviour when ``feature_names`` contains entries not in the graph.

    Returns
    -------
    pandas.DataFrame
        Long-form with columns ``name``, ``kind``, ``depth``, ``parent``,
        ``path``, ``protected_group``, ``value``, ``reference_value``,
        ``feature_count``. ``df.attrs`` carries ``agg``,
        ``reference_group``, and ``protected_order`` (reference first,
        then the rest in input order).
    """

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"shap_values must be 2D (N, F); got {arr.shape}")
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {arr.shape[1]} cols but feature_names has {len(feature_names)}"
        )

    series = _resolve_segments(protected, X, arr.shape[0])
    observed = _segment_order(series)
    if not observed:
        raise ValueError("protected must contain at least one non-NA category")
    if reference not in observed:
        raise KeyError(
            f"reference group {reference!r} not in observed protected values: {observed}"
        )
    # Stable order: reference first, then the rest in input order.
    protected_order = [reference, *(g for g in observed if g != reference)]

    matched, indices, _missing = align_features(graph, feature_names, on_unknown=on_unknown)
    name_to_idx = {name: idx for name, idx in zip(matched, indices, strict=True)}

    nodes = graph.nodes_in_order()
    base = empty_concept_frame(graph)
    feature_counts = np.zeros(len(nodes), dtype=int)
    per_sample_per_node = np.zeros((arr.shape[0], len(nodes)), dtype=float)
    for k, node in enumerate(nodes):
        feats = [f for f in graph.descendant_features(node) if f in name_to_idx]
        feature_counts[k] = len(feats)
        if not feats:
            continue
        cols = [name_to_idx[f] for f in feats]
        per_sample_per_node[:, k] = arr[:, cols].sum(axis=1)

    notna_mask = series.notna().to_numpy()
    seg_strings = series.astype(str).to_numpy()

    def _per_group(group_label: str) -> np.ndarray:
        mask = notna_mask & (seg_strings == group_label)
        if not mask.any():
            return np.full(len(nodes), np.nan, dtype=float)
        block = per_sample_per_node[mask]
        if agg == "mean_abs":
            return np.asarray(np.abs(block).mean(axis=0), dtype=float)
        if agg == "mean_signed":
            return np.asarray(block.mean(axis=0), dtype=float)
        raise ValueError(f"unknown agg {agg!r}; expected 'mean_abs' or 'mean_signed'")

    reference_values = _per_group(reference)

    rows: list[dict[str, object]] = []
    for group in protected_order:
        group_values = reference_values if group == reference else _per_group(group)
        for k in range(len(nodes)):
            ref_val = float(reference_values[k])
            grp_val = float(group_values[k])
            rows.append(
                {
                    "name": base.iloc[k]["name"],
                    "kind": base.iloc[k]["kind"],
                    "depth": base.iloc[k]["depth"],
                    "parent": base.iloc[k]["parent"],
                    "path": base.index[k],
                    "protected_group": group,
                    "value": grp_val - ref_val,
                    "reference_value": ref_val,
                    "feature_count": int(feature_counts[k]),
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["agg"] = agg
    df.attrs["reference_group"] = reference
    df.attrs["protected_order"] = protected_order
    return df
