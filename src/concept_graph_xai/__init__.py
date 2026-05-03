"""concept-graph-xai: concept-graph aware model interpretability."""

from __future__ import annotations

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics.ablation import auc_drop
from concept_graph_xai.metrics.coherence import coherence_importance
from concept_graph_xai.metrics.correlation import (
    CorrelationResult,
    feature_correlation,
    nullity_correlation,
    shap_correlation,
)
from concept_graph_xai.metrics.counts import feature_counts
from concept_graph_xai.metrics.importance import importance_sum
from concept_graph_xai.metrics.missingness import column_missing_rate, joint_missing_rate
from concept_graph_xai.metrics.utilization import utilization
from concept_graph_xai.plotting.auc_drop_map import auc_drop_map
from concept_graph_xai.plotting.coherence_importance_scatter import (
    coherence_importance_scatter,
)
from concept_graph_xai.plotting.concept_beeswarm import concept_beeswarm
from concept_graph_xai.plotting.correlation_block import correlation_block
from concept_graph_xai.plotting.joint_missing_map import joint_missing_map
from concept_graph_xai.plotting.regulatory_tag_overlay import regulatory_tag_overlay
from concept_graph_xai.plotting.sunburst import sunburst
from concept_graph_xai.plotting.utilization_map import utilization_map
from concept_graph_xai.prediction_explainer import (
    ConceptContribution,
    ConceptPredictionExplainer,
)

__version__ = "0.3.0"

__all__ = [
    "ConceptContribution",
    "ConceptGraph",
    "ConceptPredictionExplainer",
    "CorrelationResult",
    "__version__",
    "auc_drop",
    "auc_drop_map",
    "coherence_importance",
    "coherence_importance_scatter",
    "column_missing_rate",
    "concept_beeswarm",
    "correlation_block",
    "feature_correlation",
    "feature_counts",
    "importance_sum",
    "joint_missing_map",
    "joint_missing_rate",
    "nullity_correlation",
    "regulatory_tag_overlay",
    "shap_correlation",
    "sunburst",
    "utilization",
    "utilization_map",
]
