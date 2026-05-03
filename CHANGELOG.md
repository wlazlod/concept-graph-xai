# Changelog

All notable changes to **concept-graph-xai** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- **v0.4** — direction, uncertainty, single-prediction surface
  - `concept_beeswarm` — distribution of summed signed SHAP per concept.
  - `bootstrap_importance` + `signed_concept_bar` — bar chart with bootstrap confidence intervals.
  - `ConceptPredictionExplainer.waterfall(row=...)` — single-prediction waterfall rolled up to a chosen tree depth.
- **v0.5** — interactions, cohort, drift (interaction matrix, concept Sankey, segment Pareto, attribution drift).
- **v0.6** — fairness (concept-level disparity heatmap, protected-attribute API).
- **v1.0** — DAG support with optional per-edge weights; backwards compatible for tree users.

## [0.3.0] — 2026-05-03

Concept-design diagnostics release.

### Added

- Metrics: `feature_correlation`, `nullity_correlation`, `shap_correlation`, `joint_missing_rate`, `column_missing_rate`, `coherence_importance`.
- Plots: `correlation_block`, `joint_missing_map`, `coherence_importance_scatter`, `regulatory_tag_overlay`.
- Cross-cutting decisions locked:
  - **D1** Default correlation method is switchable (Spearman default).
  - **D2** `joint_missing_rate` is a standalone metric — no implicit fusion into `auc_drop`.
  - **D3** `shap` stays an optional extra (`pip install concept-graph-xai[shap]`).
  - **D4** Single-prediction surface is `ConceptPredictionExplainer`.

## [0.2.0]

Bug-fix release.

### Fixed

- `auc_drop_map` rendering empty when `skip_root=True`. Root `feature_count` was 0 with non-zero children, which Plotly silently dropped.

## [0.1.0]

Minimum viable release.

### Added

- `ConceptGraph` (tree, NetworkX-backed) with YAML / dict / NetworkX constructors.
- Metrics: `feature_counts`, `importance_sum`, `utilization`, `auc_drop` (three ablation strategies).
- Plots: `sunburst`, `utilization_map`, `auc_drop_map`.
- Adapters: `from_shap_explanation`, `from_permutation_importance`, `from_feature_importances_`.
- Tests, mypy strict, README quickstart, end-to-end notebook on the Give Me Some Credit Kaggle dataset.

[Unreleased]: https://github.com/wlazlod/concept-graph-xai/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.3.0
[0.2.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.2.0
[0.1.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.1.0
