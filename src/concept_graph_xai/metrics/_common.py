"""Internal helpers shared across metric implementations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph


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
