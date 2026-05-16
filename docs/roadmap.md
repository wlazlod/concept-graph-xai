# Roadmap

This page is the high-level roadmap, summarising shipped milestones and planned work.

## Released

### v0.1 — Minimum viable
- `ConceptGraph` (tree, NetworkX-backed) with YAML / dict / NetworkX constructors.
- Metrics: `feature_counts`, `importance_sum`, `utilization`, `auc_drop` (3 strategies).
- Plots: `sunburst`, `utilization_map`, `auc_drop_map`.
- Adapters: `from_shap_explanation`, `from_permutation_importance`, `from_feature_importances_`.
- Tests, mypy strict, README quickstart, end-to-end notebook on Give Me Some Credit.

### v0.2 — Bug-fix
- Fixed `auc_drop_map` rendering empty when `skip_root=True` (root `feature_count` was 0 with non-zero children, which Plotly silently dropped).

### v0.3 — Concept-design diagnostics
- New metrics: `feature_correlation`, `nullity_correlation`, `shap_correlation`, `joint_missing_rate`, `column_missing_rate`, `coherence_importance`.
- New plots: `correlation_block`, `joint_missing_map`, `coherence_importance_scatter`, `regulatory_tag_overlay`.
- Cross-cutting decisions locked: switchable correlation method (Spearman default), `joint_missing_rate` is a standalone metric (no implicit fusion into `auc_drop`), `shap` stays an optional extra.

### v0.4 — Direction, single-prediction, rendering polish
- `concept_violin` — per-concept horizontal violin (KDE) of summed signed SHAP per sample, branch-tinted.
- `ConceptPredictionExplainer.waterfall(row=...)` — single-prediction waterfall rolled up to a chosen tree depth.
- Rendering defaults: root concept hidden by default; `sunburst` colours by branch (hierarchical shading) when no colorscale is given; `utilization_map` subsumes the standalone feature-count sunburst; `correlation_block` stacks nested concept labels in depth-stratified rows.

### v0.5 — Uncertainty, interactions, cohort, drift (current)
- Uncertainty: `bootstrap_importance` + `signed_concept_bar` — bar chart with bootstrap confidence intervals.
- Interactions: `concept_interaction_matrix` + `concept_interaction_heatmap` — concept × concept SHAP-interaction matrix; `concept_sankey` — multi-tier SHAP flow diagram with explicit ontological grouping.
- Cohort: `segment_importance` + `segment_concept_heatmap` — concept × cohort heatmap; `concept_pareto` — faceted Lorenz curves per cohort.
- Drift: `attribution_drift` + `concept_drift_lines` — multi-period attribution monitoring; `concept_drift_delta` + `concept_drift_sunburst` — period-to-period delta sunburst.

## Planned

### v0.6 — Fairness
- `concept_disparity` + `concept_disparity_heatmap` — concept × protected-group disparity matrix.
- Protected-attribute API design pass before implementation.

### v1.0 — DAG support
- Optional per-edge weights for multi-parent concepts.
- Sankey rendering for the DAG case.
- Backwards-compatible: tree users see no change.

## Decision log

The four cross-cutting decisions locked during the v0.3 grooming session:

| ID | Decision | Locked value |
|---|---|---|
| **D1** | Default correlation method | Switchable, default = Spearman |
| **D2** | `auc_drop` realism weight | Standalone `joint_missing_rate` metric, no implicit fusion |
| **D3** | SHAP dependency posture | Optional extra (`[shap]`) |
| **D4** | Single-prediction surface | `ConceptPredictionExplainer` class |
