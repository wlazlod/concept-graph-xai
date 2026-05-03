"""Block-structured correlation matrices (P14, P15a, P17).

Every function here returns a :class:`CorrelationResult` carrying the
correlation matrix in graph-DFS feature order, the block boundaries (per
:func:`concept_graph_xai.metrics._common.block_boundaries`), per-block
``mean(|r|)`` aggregates, and the method that was used.

The plotting layer consumes this object directly via
:func:`concept_graph_xai.plotting.correlation_block.correlation_block`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import block_boundaries

CorrelationMethod = Literal["spearman", "pearson"]


@dataclass(frozen=True)
class CorrelationResult:
    """Output of every correlation metric in this module."""

    matrix: pd.DataFrame
    """Square ``DataFrame`` indexed/columned by feature name in graph order."""

    blocks: list[tuple[str, int, int]]
    """``[(concept_path, start_idx, end_idx_exclusive), ...]`` over rows/cols."""

    block_stats: pd.DataFrame
    """One row per block. Columns: ``concept_path``, ``size``, ``mean_abs``,
    ``median_abs``, ``min``, ``max``."""

    method: CorrelationMethod


def _ordered_feature_names(graph: ConceptGraph, X_columns: Sequence[str]) -> list[str]:
    """Intersect ``graph.features()`` with ``X_columns`` keeping graph order."""

    available = set(X_columns)
    return [f for f in graph.features() if f in available]


def _block_aggregates(
    matrix: np.ndarray,
    blocks: list[tuple[str, int, int]],
) -> pd.DataFrame:
    """Per-block mean/median/min/max of ``|r|`` over off-diagonal entries."""

    rows: list[dict[str, float | str | int]] = []
    for path, start, end in blocks:
        block = matrix[start:end, start:end]
        size = end - start
        if size <= 1:
            mean_abs = median_abs = float("nan")
            mn = mx = float("nan")
        else:
            iu = np.triu_indices(size, k=1)
            off = block[iu]
            abs_off = np.abs(off)
            mean_abs = float(np.nanmean(abs_off))
            median_abs = float(np.nanmedian(abs_off))
            mn = float(np.nanmin(off))
            mx = float(np.nanmax(off))
        rows.append(
            {
                "concept_path": path,
                "size": size,
                "mean_abs": mean_abs,
                "median_abs": median_abs,
                "min": mn,
                "max": mx,
            }
        )
    return pd.DataFrame(rows)


def feature_correlation(
    graph: ConceptGraph,
    X: pd.DataFrame,
    *,
    method: CorrelationMethod = "spearman",
) -> CorrelationResult:
    """Block-structured correlation matrix on feature *values* (P14).

    Diagonal blocks reveal *within-concept coherence*; off-diagonal blocks
    reveal *boundary leakage* (features in different concepts that turn out to
    be highly correlated).
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("feature_correlation requires a pandas DataFrame X")
    feats = _ordered_feature_names(graph, list(X.columns))
    if not feats:
        raise ValueError("no overlap between graph features and X columns")
    sub = X.loc[:, feats]
    matrix = sub.corr(method=method)
    blocks = block_boundaries(graph, feature_names=feats)
    block_stats = _block_aggregates(matrix.to_numpy(), blocks)
    return CorrelationResult(matrix=matrix, blocks=blocks, block_stats=block_stats, method=method)


def nullity_correlation(
    graph: ConceptGraph,
    X: pd.DataFrame,
    *,
    method: CorrelationMethod = "spearman",
) -> CorrelationResult:
    """Block-structured correlation matrix on feature *missingness* (P15a).

    Built on ``X.isna()``. A high diagonal-block value means the features in
    that concept tend to go missing together — directly relevant to the AUC
    drop "this branch is missing" scenario.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("nullity_correlation requires a pandas DataFrame X")
    feats = _ordered_feature_names(graph, list(X.columns))
    if not feats:
        raise ValueError("no overlap between graph features and X columns")
    indicators = X.loc[:, feats].isna().astype(float)
    indicators = indicators.loc[:, indicators.std() > 0]
    if indicators.shape[1] == 0:
        empty = pd.DataFrame(np.zeros((len(feats), len(feats))), index=feats, columns=feats)
        blocks = block_boundaries(graph, feature_names=feats)
        return CorrelationResult(
            matrix=empty,
            blocks=blocks,
            block_stats=_block_aggregates(empty.to_numpy(), blocks),
            method=method,
        )
    raw = indicators.corr(method=method)
    matrix = raw.reindex(index=feats, columns=feats).fillna(0.0)
    blocks = block_boundaries(graph, feature_names=feats)
    block_stats = _block_aggregates(matrix.to_numpy(), blocks)
    return CorrelationResult(matrix=matrix, blocks=blocks, block_stats=block_stats, method=method)


def shap_correlation(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    *,
    method: CorrelationMethod = "spearman",
) -> CorrelationResult:
    """Block-structured correlation of *SHAP values* across samples (P17).

    Two raw-uncorrelated features can still be SHAP-redundant: diagonal blocks
    near 1 indicate features inside a concept push the model in the same way;
    off-diagonal blocks near 1 indicate the model treats different concepts as
    substitutes.
    """

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"shap_values must be 2D (N, F); got shape {arr.shape}")
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {arr.shape[1]} features, feature_names has {len(feature_names)}"
        )
    df = pd.DataFrame(arr, columns=list(feature_names))
    feats = _ordered_feature_names(graph, list(df.columns))
    sub = df.loc[:, feats]
    matrix = sub.corr(method=method)
    blocks = block_boundaries(graph, feature_names=feats)
    block_stats = _block_aggregates(matrix.to_numpy(), blocks)
    return CorrelationResult(matrix=matrix, blocks=blocks, block_stats=block_stats, method=method)
