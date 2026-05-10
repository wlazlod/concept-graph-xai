"""Metric layer: returns tidy ``pandas.DataFrame`` keyed by concept path.

Every public function in this layer is plot-agnostic. The plotting layer
consumes these DataFrames via :class:`~concept_graph_xai.graph.ConceptGraph`.
"""

from concept_graph_xai.metrics.ablation import auc_drop
from concept_graph_xai.metrics.bootstrap import bootstrap_importance
from concept_graph_xai.metrics.coherence import coherence_importance
from concept_graph_xai.metrics.correlation import (
    CorrelationResult,
    feature_correlation,
    nullity_correlation,
    shap_correlation,
)
from concept_graph_xai.metrics.counts import feature_counts
from concept_graph_xai.metrics.importance import importance_sum
from concept_graph_xai.metrics.interaction import concept_interaction_matrix
from concept_graph_xai.metrics.missingness import column_missing_rate, joint_missing_rate
from concept_graph_xai.metrics.utilization import utilization

__all__ = [
    "CorrelationResult",
    "auc_drop",
    "bootstrap_importance",
    "coherence_importance",
    "column_missing_rate",
    "concept_interaction_matrix",
    "feature_correlation",
    "feature_counts",
    "importance_sum",
    "joint_missing_rate",
    "nullity_correlation",
    "shap_correlation",
    "utilization",
]
