# API reference

Auto-generated from source-code docstrings via
[`mkdocstrings`](https://mkdocstrings.github.io/). Metric functions return a tidy
`pandas.DataFrame` indexed by the concept's `/`-joined path; plot functions return a
`plotly.graph_objects.Figure`.

## ConceptGraph

The tree-shaped concept graph. Constructors from dict / YAML / NetworkX; deterministic
DFS traversal.

::: concept_graph_xai.graph.ConceptGraph

::: concept_graph_xai.graph.NodeView

## IO

YAML load/dump for the nested-dict graph format — the plumbing behind
`ConceptGraph.from_yaml` / `ConceptGraph.to_yaml`.

::: concept_graph_xai.io.load_yaml

::: concept_graph_xai.io.dump_yaml

## Adapters

Convert SHAP, sklearn permutation results, and `model.feature_importances_` into the
canonical `(values, feature_names)` tuple.

::: concept_graph_xai.adapters.shap.from_shap_explanation

::: concept_graph_xai.adapters.sklearn_perm.from_permutation_importance

::: concept_graph_xai.adapters.tree_native.from_feature_importances_

## Counts

How many features live under each concept?

::: concept_graph_xai.metrics.counts.feature_counts

## Importance

How much importance does each concept aggregate?

::: concept_graph_xai.metrics.importance.importance_sum

## Bootstrap

Percentile confidence intervals on per-concept importance — is the ranking
statistically separable?

::: concept_graph_xai.metrics.bootstrap.bootstrap_importance

## Utilization

Which concepts does the model actually use?

::: concept_graph_xai.metrics.utilization.utilization

## Interaction

Concept × concept aggregation of the feature-level SHAP-interaction tensor.

::: concept_graph_xai.metrics.interaction.concept_interaction_matrix

## Ablation

How much performance is lost when a concept's data is missing? Three strategies.

::: concept_graph_xai.metrics.ablation.auc_drop

## Correlation

Are concepts internally coherent? Do they go missing together? Do features look
substitutable to the model?

::: concept_graph_xai.metrics.correlation.CorrelationResult

::: concept_graph_xai.metrics.correlation.feature_correlation

::: concept_graph_xai.metrics.correlation.nullity_correlation

::: concept_graph_xai.metrics.correlation.shap_correlation

## Missingness

How often does a feature / a whole concept go missing?

::: concept_graph_xai.metrics.missingness.column_missing_rate

::: concept_graph_xai.metrics.missingness.joint_missing_rate

## Coherence

Are concepts well-designed (coherent + important)?

::: concept_graph_xai.metrics.coherence.coherence_importance

## Segments

Per-segment per-concept SHAP aggregation for cohort analysis.

::: concept_graph_xai.metrics.segment.segment_importance

## Disparity

Per-concept SHAP gap of each protected group against a reference group.

::: concept_graph_xai.metrics.disparity.concept_disparity

## Drift

Per-period concept SHAP aggregation for drift monitoring: a long-form per-period
table and a two-period delta table.

::: concept_graph_xai.metrics.drift.attribution_drift

::: concept_graph_xai.metrics.drift.concept_drift_delta

## Plotting

All plots return `plotly.graph_objects.Figure`. Static PNG via the `[png]` extra
(kaleido).

::: concept_graph_xai.plotting.sunburst.sunburst

::: concept_graph_xai.plotting.utilization_map.utilization_map

::: concept_graph_xai.plotting.signed_concept_bar.signed_concept_bar

::: concept_graph_xai.plotting.concept_interaction_heatmap.concept_interaction_heatmap

::: concept_graph_xai.plotting.concept_sankey.concept_sankey

::: concept_graph_xai.plotting.concept_violin.concept_violin

::: concept_graph_xai.plotting.auc_drop_map.auc_drop_map

::: concept_graph_xai.plotting.correlation_block.correlation_block

::: concept_graph_xai.plotting.joint_missing_map.joint_missing_map

::: concept_graph_xai.plotting.coherence_importance_scatter.coherence_importance_scatter

::: concept_graph_xai.plotting.regulatory_tag_overlay.regulatory_tag_overlay

::: concept_graph_xai.plotting.segment_concept_heatmap.segment_concept_heatmap

::: concept_graph_xai.plotting.concept_pareto.concept_pareto

::: concept_graph_xai.plotting.concept_disparity_heatmap.concept_disparity_heatmap

::: concept_graph_xai.plotting.concept_drift_lines.concept_drift_lines

::: concept_graph_xai.plotting.concept_drift_sunburst.concept_drift_sunburst

## Prediction explainer

Single-prediction explanations rolled up to concept level, headlined by the
concept waterfall.

::: concept_graph_xai.prediction_explainer.ConceptPredictionExplainer

::: concept_graph_xai.prediction_explainer.ConceptContribution
