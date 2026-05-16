"""Bootstrap confidence intervals on per-concept SHAP (P2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import (
    align_features,
    empty_concept_frame,
    per_sample_per_concept,
)

BootstrapAgg = Literal["mean_signed", "mean_abs"]


def bootstrap_importance(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    *,
    n_bootstrap: int = 200,
    ci: float = 0.95,
    random_state: int | None = None,
    agg: BootstrapAgg = "mean_signed",
    on_unknown: str = "warn",
    signed: bool | None = None,
) -> pd.DataFrame:
    """Bootstrap-resampled per-concept SHAP with percentile CI.

    For each concept, the per-sample value is the sum of SHAP across the
    concept's descendant features. The statistic is the mean of those
    per-sample values across the held-out set. The bootstrap resamples row
    indices with replacement ``n_bootstrap`` times and reports the mean
    plus percentile confidence-interval bounds.

    Parameters
    ----------
    graph:
        The ConceptGraph.
    feature_names:
        Names matching the columns of ``shap_values``.
    shap_values:
        Per-sample SHAP values of shape ``(N, F)``.
    n_bootstrap:
        Number of resamples (default 200).
    ci:
        Confidence level in ``(0, 1)``. Default ``0.95`` → 2.5% / 97.5%.
    random_state:
        Seed for the resampler.
    agg:
        ``"mean_signed"`` (default) uses the signed per-sample sum
        (cancellation can shrink the magnitude). ``"mean_abs"`` uses
        ``sum(|SHAP|)`` and is always non-negative. Naming matches the
        v0.5/v0.6 family (``segment_importance``, ``concept_disparity``,
        ``attribution_drift``, ``concept_interaction_matrix``).
    on_unknown:
        Behaviour when ``feature_names`` contains entries not in the graph.
    signed:
        **Deprecated.** Pass ``agg="mean_signed"`` (when ``signed=True``) or
        ``agg="mean_abs"`` (when ``signed=False``). Kept as an alias for
        v0.6 callers; will be removed in a future release.

    Returns
    -------
    pandas.DataFrame
        Indexed by concept-path with columns ``name``, ``kind``, ``depth``,
        ``parent``, the value column (``mean_signed_shap`` or
        ``mean_abs_shap``), ``ci_lo``, ``ci_hi``, ``feature_count``.
    """

    if signed is not None:
        import warnings

        warnings.warn(
            "signed= is deprecated; pass agg='mean_signed' or agg='mean_abs' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        agg = "mean_signed" if signed else "mean_abs"
    if agg not in ("mean_signed", "mean_abs"):
        raise ValueError(f"unknown agg {agg!r}; expected 'mean_signed' or 'mean_abs'")

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"shap_values must be 2D (N, F); got {arr.shape}")
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {arr.shape[1]} cols but feature_names has {len(feature_names)}"
        )
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1); got {ci}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1; got {n_bootstrap}")

    matched, indices, _missing = align_features(graph, feature_names, on_unknown=on_unknown)
    name_to_idx = {name: idx for name, idx in zip(matched, indices, strict=True)}

    nodes = graph.nodes_in_order()
    n_samples, _ = arr.shape
    # For mean_abs we want |SHAP| summed (cancellation does NOT shrink), so
    # we feed the absolute array into the per-sample helper. For mean_signed
    # we feed the signed array.
    source = np.abs(arr) if agg == "mean_abs" else arr
    per_sample_per_node, feature_counts = per_sample_per_concept(graph, source, name_to_idx)

    rng = np.random.default_rng(random_state)
    boot_means = np.empty((n_bootstrap, len(nodes)), dtype=float)
    for b in range(n_bootstrap):
        sampled = rng.integers(0, n_samples, size=n_samples)
        boot_means[b] = per_sample_per_node[sampled].mean(axis=0)

    alpha = (1.0 - ci) / 2.0
    ci_lo = np.percentile(boot_means, 100.0 * alpha, axis=0)
    ci_hi = np.percentile(boot_means, 100.0 * (1.0 - alpha), axis=0)
    mean_estimate = boot_means.mean(axis=0)

    df = empty_concept_frame(graph)
    value_col = "mean_signed_shap" if agg == "mean_signed" else "mean_abs_shap"
    df[value_col] = mean_estimate
    df["ci_lo"] = ci_lo
    df["ci_hi"] = ci_hi
    df["feature_count"] = feature_counts
    df.attrs["ci"] = ci
    df.attrs["n_bootstrap"] = n_bootstrap
    df.attrs["agg"] = agg
    # Back-compat: keep df.attrs["signed"] for one release.
    df.attrs["signed"] = agg == "mean_signed"
    return df
