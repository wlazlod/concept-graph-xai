"""Plotly-based plotting layer.

The plotting layer is decoupled from the metric layer: it consumes a
:class:`~concept_graph_xai.graph.ConceptGraph` plus a tidy DataFrame produced
by one of the metric functions.
"""

from concept_graph_xai.plotting.auc_drop_map import auc_drop_map
from concept_graph_xai.plotting.coherence_importance_scatter import (
    coherence_importance_scatter,
)
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

__all__ = [
    "auc_drop_map",
    "coherence_importance_scatter",
    "concept_interaction_heatmap",
    "concept_pareto",
    "concept_sankey",
    "concept_violin",
    "correlation_block",
    "joint_missing_map",
    "regulatory_tag_overlay",
    "segment_concept_heatmap",
    "signed_concept_bar",
    "sunburst",
    "utilization_map",
]
