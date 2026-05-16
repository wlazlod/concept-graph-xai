"""Internal helpers shared across metric implementations."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph


def deprecated_kwarg_or[D, R](
    deprecated_value: D | None,
    current_value: R,
    *,
    old: str,
    new: str,
    transform: Callable[[D], R] | None = None,
    stacklevel: int = 3,
) -> R:
    """Pick between a deprecated kwarg and the current one, warning when used.

    If ``deprecated_value`` is ``None`` the caller didn't pass the
    deprecated kwarg and ``current_value`` is returned unchanged. Otherwise
    a ``DeprecationWarning`` is emitted and (a possibly transformed)
    ``deprecated_value`` wins, mirroring the original "deprecated arg
    overrides" semantics each call site implemented inline.

    Used by every kwarg-only deprecation alias in the library
    (``signed=`` on ``bootstrap_importance``, ``include_root=`` on the
    four heatmap plots, ``top_k=`` on ``concept_drift_lines``).
    """

    if deprecated_value is None:
        return current_value
    suggestion = f"{new}=not {old}" if transform else f"{new}"
    warnings.warn(
        f"{old}= is deprecated; pass {suggestion} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    if transform is not None:
        return transform(deprecated_value)
    # Callers that pass no transform implicitly assert D == R.
    return cast(R, deprecated_value)


def per_sample_per_concept(
    graph: ConceptGraph,
    arr: np.ndarray,
    name_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample summed-over-descendants values for every graph node.

    Returns ``(values, feature_counts)`` where ``values`` has shape
    ``(arr.shape[0], len(graph))`` and ``feature_counts`` has shape
    ``(len(graph),)``. Both follow ``graph.nodes_in_order()`` ordering.

    For each node, the per-sample value is
    ``sum_{f in node.descendants ∩ name_to_idx} arr[:, name_to_idx[f]]``.
    Nodes with no descendants in ``name_to_idx`` have a zero column and a
    zero feature count.

    Used by every "per-group per-concept" metric (``segment_importance``,
    ``concept_disparity``, ``attribution_drift``, ``bootstrap_importance``)
    so the inner loop lives in one place.
    """

    nodes = graph.nodes_in_order()
    values = np.zeros((arr.shape[0], len(nodes)), dtype=float)
    feature_counts = np.zeros(len(nodes), dtype=int)
    for k, node in enumerate(nodes):
        feats = [f for f in graph.descendant_features(node) if f in name_to_idx]
        feature_counts[k] = len(feats)
        if not feats:
            continue
        cols = [name_to_idx[f] for f in feats]
        values[:, k] = arr[:, cols].sum(axis=1)
    return values, feature_counts


def resolve_grouping(
    grouping: pd.Series | str,
    X: pd.DataFrame | None,
    n_rows: int,
    *,
    param_name: str = "grouping",
) -> pd.Series:
    """Coerce a Series-or-column-name grouping vector to an index-aligned Series.

    Used by every metric that takes a row-level categorical vector
    (segments, protected attributes, periods, cohorts).
    """

    if isinstance(grouping, str):
        if X is None:
            raise ValueError(
                f"{param_name} is a column-name string; pass X=... so we can pull the column"
            )
        if grouping not in X.columns:
            raise KeyError(f"{param_name} column {grouping!r} not in X.columns")
        series = X[grouping]
    elif isinstance(grouping, pd.Series):
        series = grouping
    else:
        raise TypeError(
            f"{param_name} must be a pandas Series or column-name str, "
            f"got {type(grouping).__name__}"
        )

    if len(series) != n_rows:
        raise ValueError(f"{param_name} has {len(series)} rows but the data has {n_rows}")
    return series.reset_index(drop=True)


def grouping_order(series: pd.Series) -> list[str]:
    """Stable category ordering: categorical-ordered if available, else first-seen."""

    if isinstance(series.dtype, pd.CategoricalDtype) and series.cat.ordered:
        return [str(c) for c in series.cat.categories]
    seen: list[str] = []
    for value in series.dropna().astype(str):
        if value not in seen:
            seen.append(value)
    return seen


def aggregate_per_feature(
    values: np.ndarray,
    *,
    signed: bool = False,
    agg: str = "mean",
) -> np.ndarray:
    """Reduce a 2D per-sample importance array to a 1D per-feature aggregate.

    Parameters
    ----------
    values:
        Either ``(F,)`` (already aggregated) or ``(N, F)``.
    signed:
        If ``False`` (default), takes ``np.abs`` of values before aggregating.
        If ``True``, keeps the sign — useful for SHAP-flavoured signed sums.
    agg:
        One of ``{"mean", "sum"}``.
    """

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return np.asarray(np.abs(arr) if not signed else arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D values, got rank {arr.ndim}")
    if not signed:
        arr = np.abs(arr)
    if agg == "mean":
        return np.asarray(arr.mean(axis=0), dtype=float)
    if agg == "sum":
        return np.asarray(arr.sum(axis=0), dtype=float)
    raise ValueError(f"unknown agg {agg!r}; expected 'mean' or 'sum'")


def aligned_index_map(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    *,
    on_unknown: str = "warn",
) -> dict[str, int]:
    """Return ``{graph_feature: index_in_feature_names}`` for matched features.

    Convenience wrapper over :func:`align_features` for callers that only
    need the lookup dict (and not the matched-names list or the unknown-
    names list). Used by every metric that builds a per-sample-per-concept
    aggregate from a (N, F) array — five call sites were each constructing
    this dict inline.
    """

    matched, indices, _missing = align_features(graph, feature_names, on_unknown=on_unknown)
    return {name: idx for name, idx in zip(matched, indices, strict=True)}


def align_features(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    *,
    on_unknown: str = "warn",
) -> tuple[list[str], list[int], list[str]]:
    """Return ``(graph_features_in_input_order, indices, missing_in_graph)``.

    The first list contains feature names that appear in *both* the graph and
    ``feature_names``, ordered by their position in ``feature_names``. The
    second list is their indices into ``feature_names``. The third list reports
    feature names in ``feature_names`` not present in the graph.

    ``on_unknown`` controls behaviour when ``feature_names`` contains entries
    not present in the graph: ``"warn"`` (default), ``"ignore"`` or ``"raise"``.
    """

    graph_features = set(graph.features())
    matched: list[str] = []
    indices: list[int] = []
    missing: list[str] = []
    for idx, name in enumerate(feature_names):
        if name in graph_features:
            matched.append(name)
            indices.append(idx)
        else:
            missing.append(name)

    if missing and on_unknown == "raise":
        raise KeyError(f"features not in graph: {missing!r}")
    if missing and on_unknown == "warn":
        import warnings

        warnings.warn(
            f"{len(missing)} features in input are not present in the graph: "
            f"{missing[:3]}{'...' if len(missing) > 3 else ''}",
            stacklevel=3,
        )
    return matched, indices, missing


def block_boundaries(
    graph: ConceptGraph,
    feature_names: Sequence[str] | None = None,
) -> list[tuple[str, int, int]]:
    """Return ``(concept_path, start_idx, end_idx)`` for every concept whose
    feature descendants form a contiguous block when features are listed in
    ``graph.features()`` order (or in the explicitly supplied ``feature_names``
    if it equals that order).

    Used by the §H block-correlation plots to draw separator lines and to
    annotate per-block aggregate statistics. ``end_idx`` is exclusive.

    The returned list is depth-ordered: top-level concepts first, then deeper
    ones. Single-feature concepts (e.g. ``Age`` wrapping ``age``) are included
    with ``end_idx - start_idx == 1``.
    """

    feats = list(feature_names) if feature_names is not None else graph.features()
    feat_to_idx = {name: i for i, name in enumerate(feats)}
    blocks: list[tuple[str, int, int]] = []
    for node in graph.nodes_in_order():
        if graph.kind(node) == "feature":
            continue
        descendants = [f for f in graph.descendant_features(node) if f in feat_to_idx]
        if not descendants:
            continue
        idxs = sorted(feat_to_idx[f] for f in descendants)
        if idxs == list(range(idxs[0], idxs[-1] + 1)):
            path = "/".join(graph.path(node))
            blocks.append((path, idxs[0], idxs[-1] + 1))
    return blocks


def empty_concept_frame(graph: ConceptGraph) -> pd.DataFrame:
    """Build a baseline DataFrame indexed by concept-path string for every node."""

    paths: list[str] = ["/".join(graph.path(n)) for n in graph.nodes_in_order()]
    df = pd.DataFrame(
        {
            "name": graph.nodes_in_order(),
            "kind": [graph.kind(n) for n in graph.nodes_in_order()],
            "depth": [len(graph.path(n)) - 1 for n in graph.nodes_in_order()],
            "parent": [graph.parent_of(n) or "" for n in graph.nodes_in_order()],
        },
        index=pd.Index(paths, name="path"),
    )
    return df
