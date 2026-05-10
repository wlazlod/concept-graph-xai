"""Bootstrap confidence intervals on per-concept signed SHAP (P2)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import align_features, empty_concept_frame


def bootstrap_importance(
    graph: ConceptGraph,
    feature_names: Sequence[str],
    shap_values: np.ndarray,
    *,
    n_bootstrap: int = 200,
    ci: float = 0.95,
    random_state: int | None = None,
    signed: bool = True,
    on_unknown: str = "warn",
) -> pd.DataFrame:
    """Bootstrap-resampled per-concept summed SHAP with percentile CI.

    For each concept, the per-sample value is the sum of SHAP across the
    concept's descendant features (signed by default). The statistic is the
    mean of those per-sample values across the held-out set. The bootstrap
    resamples row indices with replacement ``n_bootstrap`` times and reports
    the mean and the symmetric percentile CI bounds for each concept.

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
    signed:
        If ``True`` (default), the per-sample concept value is the *signed*
        sum across descendants (so cancellation can shrink it). If ``False``,
        the per-sample value is the sum of ``|SHAP|`` (always non-negative).
    on_unknown:
        Behaviour when ``feature_names`` contains entries not present in the
        graph: ``"warn"`` (default), ``"ignore"``, ``"raise"``.

    Returns
    -------
    pandas.DataFrame
        Indexed by concept-path with columns ``name``, ``kind``, ``depth``,
        ``parent``, ``mean_signed_shap`` (or ``mean_abs_shap`` when
        ``signed=False``), ``ci_lo``, ``ci_hi``, ``feature_count``.
    """

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
    per_sample_per_node = np.zeros((n_samples, len(nodes)), dtype=float)
    feature_counts = np.zeros(len(nodes), dtype=int)
    for k, node in enumerate(nodes):
        feats = [f for f in graph.descendant_features(node) if f in name_to_idx]
        feature_counts[k] = len(feats)
        if not feats:
            continue
        idxs = [name_to_idx[f] for f in feats]
        sub = arr[:, idxs]
        if not signed:
            sub = np.abs(sub)
        per_sample_per_node[:, k] = sub.sum(axis=1)

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
    value_col = "mean_signed_shap" if signed else "mean_abs_shap"
    df[value_col] = mean_estimate
    df["ci_lo"] = ci_lo
    df["ci_hi"] = ci_hi
    df["feature_count"] = feature_counts
    df.attrs["ci"] = ci
    df.attrs["n_bootstrap"] = n_bootstrap
    df.attrs["signed"] = signed
    return df
