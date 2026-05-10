# Changelog

All notable changes to **concept-graph-xai** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- **v0.5** — direction & uncertainty (`bootstrap_importance` + `signed_concept_bar` — bar chart with bootstrap confidence intervals); interactions, cohort, drift (interaction matrix, concept Sankey, segment Pareto, attribution drift).
- **v0.6** — fairness (concept-level disparity heatmap, protected-attribute API).
- **v1.0** — DAG support with optional per-edge weights; backwards compatible for tree users.

## [0.4.0] — 2026-05-09

Local-explanation surface and rendering-default cleanups.

### Added

- `concept_violin` — per-concept horizontal violin (KDE) of summed signed SHAP across samples, tinted by the concept's top-level branch with hierarchical shading. Replaces the earlier strip / box prototype: width at each x is the sample density, optional inner box and mean line, raw points on by default for outliers only (`points="outliers"`).
- `ConceptPredictionExplainer.waterfall(row=...)` — single-prediction waterfall rolled up to a chosen tree depth.
- `sunburst()` gains `color_by={"auto","branch","value","none"}` and `branch_palette` knobs. `color_by="branch"` colours each top-level concept (and its descendants) with one categorical hue from the palette.
- `hide_root: bool = True` keyword on every sunburst plot (`sunburst`, `utilization_map`, `auc_drop_map`, `joint_missing_map`, `regulatory_tag_overlay`).

### Changed

- **Breaking (rendering only):** sunburst plots now default to `hide_root=True`. The root concept always rolls up to 100% / total, so the centre sector carries no signal; hiding it puts the first-level concepts at the centre. Pass `hide_root=False` for the previous look.
- **Breaking (rendering only):** `sunburst()` defaults to branch-categorical coloring when no `colorscale` is supplied. Sector area already encodes the value (e.g. `importance_sum`), so the colour channel identifies which branch a concept belongs to.
- **Hierarchical branch shading.** Branch coloring now lightens shades with depth: the top-level concept is the saturated palette base; sub-concepts and leaves are progressively blended toward white (capped at 55% so they stay visible). This makes the within-branch hierarchy readable on top of the cross-branch hue separation.
- **`utilization_map` subsumes the standalone feature-count sunburst.** Default coloring is now branch-hierarchical for used sectors and grey for unused ones. Sector area still uses `feature_count`. A single chart now answers "how big is each concept *and* which ones does the model use." Pass `used_color="<css>"` to fall back to the legacy single-colour fill.
- **`correlation_block` no longer overlaps nested concept labels.** Block labels are stacked in depth-stratified rows below the heatmap (top-level branches in the bottom row, sub-concepts closer to the heatmap), and the y-axis range and bottom margin grow with tree depth so labels fit cleanly.

### Notebook

- The example notebook now declares two phantom branches (`Behaviour/WebActivity` and a top-level `AlternativeData`) whose features are not present in `X`, so the `utilization_map` plot demonstrates the grey-out for declared-but-unmodelled concepts.
- The redundant standalone `sunburst(graph, feature_counts(...))` cell has been removed; `utilization_map` now serves the same structural purpose plus the utilization overlay.

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

[Unreleased]: https://github.com/wlazlod/concept-graph-xai/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.4.0
[0.3.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.3.0
[0.2.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.2.0
[0.1.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.1.0
