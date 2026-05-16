"""concept-graph-xai: concept-graph aware model interpretability."""

from __future__ import annotations

from concept_graph_xai.graph import ConceptGraph
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
from concept_graph_xai.metrics.drift import attribution_drift, concept_drift_delta
from concept_graph_xai.metrics.importance import importance_sum
from concept_graph_xai.metrics.interaction import concept_interaction_matrix
from concept_graph_xai.metrics.missingness import column_missing_rate, joint_missing_rate
from concept_graph_xai.metrics.segment import segment_importance
from concept_graph_xai.metrics.utilization import utilization
from concept_graph_xai.plotting.auc_drop_map import auc_drop_map
from concept_graph_xai.plotting.coherence_importance_scatter import (
    coherence_importance_scatter,
)
from concept_graph_xai.plotting.concept_drift_lines import concept_drift_lines
from concept_graph_xai.plotting.concept_drift_sunburst import concept_drift_sunburst
from concept_graph_xai.plotting.concept_interaction_heatmap import concept_interaction_heatmap
from concept_graph_xai.plotting.concept_pareto import concept_pareto
from concept_graph_xai.plotting.concept_sankey import concept_sankey
from concept_graph_xai.plotting.concept_violin import concept_violin
from concept_graph_xai.plotting.correlation_block import correlation_block
from concept_graph_xai.plotting.joint_missing_map import joint_missing_map
from concept_graph_xai.plotting.regulatory_tag_overlay import regulatory_tag_overlay
from concept_graph_xai.plotting.segment_concept_heatmap import segment_concept_heatmap
from concept_graph_xai.plotting.signed_concept_bar import signed_concept_bar
from concept_graph_xai.plotting.sunburst import sunburst
from concept_graph_xai.plotting.utilization_map import utilization_map
from concept_graph_xai.prediction_explainer import (
    ConceptContribution,
    ConceptPredictionExplainer,
)

__version__ = "0.5.0"

__all__ = [
    "ConceptContribution",
    "ConceptGraph",
    "ConceptPredictionExplainer",
    "CorrelationResult",
    "__version__",
    "attribution_drift",
    "auc_drop",
    "auc_drop_map",
    "bootstrap_importance",
    "coherence_importance",
    "coherence_importance_scatter",
    "column_missing_rate",
    "concept_drift_delta",
    "concept_drift_lines",
    "concept_drift_sunburst",
    "concept_interaction_heatmap",
    "concept_interaction_matrix",
    "concept_pareto",
    "concept_sankey",
    "concept_violin",
    "correlation_block",
    "feature_correlation",
    "feature_counts",
    "importance_sum",
    "joint_missing_map",
    "joint_missing_rate",
    "nullity_correlation",
    "regulatory_tag_overlay",
    "segment_concept_heatmap",
    "segment_importance",
    "shap_correlation",
    "signed_concept_bar",
    "sunburst",
    "utilization",
    "utilization_map",
]
