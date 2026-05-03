"""Concept-coherence vs concept-importance diagnostic (P16).

Joins the within-block ``mean(|r|)`` from :func:`feature_correlation` with the
summed importance from :func:`importance_sum` into a single DataFrame, one row
per concept (root included). The plotting layer renders this as a quadrant
scatter that drives the "is this tree well-shaped?" conversation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics.correlation import CorrelationMethod, feature_correlation
from concept_graph_xai.metrics.importance import importance_sum

CoherenceQuadrant = Literal[
    "well_designed",  # high coherence + high importance
    "kitchen_sink",  # low coherence + high importance
    "redundant",  # high coherence + low importance
    "noise",  # low coherence + low importance
]


def coherence_importance(
    graph: ConceptGraph,
    X: pd.DataFrame,
    feature_names: Sequence[str],
    importances: np.ndarray,
    *,
    method: CorrelationMethod = "spearman",
    coherence_threshold: float | None = None,
    importance_threshold: float | None = None,
) -> pd.DataFrame:
    """Per-concept coherence × importance table.

    Parameters
    ----------
    graph:
        ConceptGraph.
    X:
        Feature matrix used to compute within-block correlation.
    feature_names, importances:
        Inputs to :func:`importance_sum`. Per-sample (N, F) or per-feature (F,).
    method:
        Correlation method passed to :func:`feature_correlation`.
    coherence_threshold:
        Quadrant boundary on the coherence axis. Defaults to the median across
        concepts.
    importance_threshold:
        Quadrant boundary on the importance axis. Defaults to the median across
        concepts.

    Returns
    -------
    pandas.DataFrame
        One row per concept (root included). Columns include ``coherence``,
        ``importance_sum``, ``quadrant``, plus all the structural columns from
        :func:`empty_concept_frame`.
    """

    corr = feature_correlation(graph, X, method=method)
    block_lookup = corr.block_stats.set_index("concept_path")["mean_abs"].to_dict()

    imp_df = importance_sum(graph, feature_names, importances).copy()
    coherence: list[float] = []
    for path in imp_df.index:
        coherence.append(float(block_lookup.get(path, np.nan)))
    imp_df["coherence"] = coherence

    coh = np.asarray(imp_df["coherence"], dtype=float)
    imp = np.asarray(imp_df["importance_sum"], dtype=float)
    valid_coh = coh[~np.isnan(coh)]
    coh_thr = (
        float(np.median(valid_coh))
        if coherence_threshold is None and valid_coh.size > 0
        else float(coherence_threshold or 0.0)
    )
    valid_imp = imp[~np.isnan(imp)]
    imp_thr = (
        float(np.median(valid_imp))
        if importance_threshold is None and valid_imp.size > 0
        else float(importance_threshold or 0.0)
    )

    quadrants: list[str] = []
    for c, i in zip(coh, imp, strict=True):
        if np.isnan(c) or np.isnan(i):
            quadrants.append("undefined")
        elif c >= coh_thr and i >= imp_thr:
            quadrants.append("well_designed")
        elif c < coh_thr and i >= imp_thr:
            quadrants.append("kitchen_sink")
        elif c >= coh_thr and i < imp_thr:
            quadrants.append("redundant")
        else:
            quadrants.append("noise")
    imp_df["quadrant"] = quadrants
    imp_df.attrs["coherence_threshold"] = coh_thr
    imp_df.attrs["importance_threshold"] = imp_thr
    imp_df.attrs["method"] = method
    return imp_df
