# Changelog

All notable changes to **concept-graph-xai** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- **v1.0** — DAG support with optional per-edge weights; backwards compatible for tree users.

### Possible v0.6.x follow-ups (filed; no commitment)

- Intersection protected attributes (`gender × age_band`).
- First-class `ProtectedAttribute` object on the graph with per-node sensitive-flag metadata.
- Multiple-reference baselines (compare every group against the population mean instead of one group).

## [0.6.0] — 2026-05-16

Fairness — concept-level disparity vs a reference protected group.

### Added

- `concept_disparity(graph, feature_names, shap_values, protected, *, reference, X=None, agg="mean_abs"|"mean_signed")` — per-concept additive SHAP gap (`value_group - value_reference`) per protected group. `protected` accepts a `pd.Series` aligned to the rows or a column-name string (with `X` provided); `reference` is the label of the baseline group. Returns long-form DataFrame with columns `name, kind, depth, parent, path, protected_group, value, reference_value, feature_count` and `df.attrs["reference_group"]` / `protected_order`. The reference group's row is kept in the output with `value=0` so the heatmap can show it as the visible baseline.
- `concept_disparity_heatmap(graph, df)` — concept × protected-group heatmap with a diverging `RdBu` palette centred at 0. Default sort puts the most disparate concepts (max `|gap|`) at the top; `include_reference=False` drops the all-zero reference column for a tighter chart. Title auto-includes the reference label.

### Design

- **Protected-attribute API design pass** (`PROPOSALS.md` §9 + §12) locked at **minimal**: a single attribute per call, mirroring the `segment_importance` shape. Intersections, graph-level metadata, and multiple-reference baselines are intentionally deferred until a real request lands.

### Notebook

- New **Part I — Fairness** section. Synthesises a protected attribute from `X_test["age"]` (`senior >= 65` vs `non_senior < 65`, reference = `non_senior`) and renders `concept_disparity_heatmap` on the held-out SHAP values. Outline and PNG export list updated.

## [0.5.0] — 2026-05-16

Uncertainty, interactions, cohort analysis, and drift monitoring.

### Added — uncertainty

- `bootstrap_importance(graph, feature_names, shap_values, n_bootstrap=200, ci=0.95, signed=True)` — resamples row indices with replacement, recomputes per-concept summed (signed or absolute) SHAP per resample, returns the mean plus percentile confidence-interval bounds.
- `signed_concept_bar(graph, df)` — horizontal bar chart of per-concept mean signed SHAP with error bars from `ci_lo / ci_hi`. Branch-coloured via `branch_colors`, sorted by `|mean|` desc, symmetric x-axis with a dashed zero line.

### Added — interactions

- `concept_interaction_matrix(graph, feature_names, shap_interaction_values, agg="mean_abs"|"mean_signed")` — aggregates a `(N, F, F)` SHAP-interaction tensor (from `shap.TreeExplainer.shap_interaction_values`) into a symmetric concept × concept DataFrame. Diagonal cells carry within-concept self-interaction.
- `concept_interaction_heatmap(matrix)` — `go.Heatmap` over the concept × concept matrix. Sequential `Reds` for `mean_abs`, diverging `RdBu` (centred at 0) for `mean_signed`. Top-K off-diagonal cells annotated by value.
- `concept_sankey(graph, feature_names, shap_values, max_features_per_concept=None)` — multi-tier SHAP-flow Sankey that walks the **full** concept hierarchy: features → sub-concepts → … → top-level concepts → +/- outcome. Explicit per-node `(x, y)` so layout follows ontological grouping (siblings adjacent vertically, parents to the right of their children). Within-concept cancellation visibly narrows the band as you move toward the outcome side.

### Added — cohort analysis

- `segment_importance(graph, feature_names, shap_values, segments, *, X=None, agg="mean_abs"|"mean_signed")` — per-segment per-concept SHAP aggregate (long-form). `segments` accepts either a `pd.Series` aligned to the rows or a column-name string (then `X` must be provided).
- `segment_concept_heatmap(graph, df)` — concept × segment `go.Heatmap`. Default sort by max-across-segments puts the most cohort-discriminating concepts at the top. Categorical / first-seen segment order preserved via `df.attrs["segment_order"]`.
- `concept_pareto(graph, df)` — per-cohort Lorenz / Pareto curves of concept-importance concentration. One `go.Scatter` per cohort plus a dashed 45° equality reference; cohorts with all-zero importance are skipped silently.

### Added — drift

- `attribution_drift(graph, periods, *, agg="mean_abs"|"mean_signed")` — per-period per-concept SHAP aggregate (long-form). `periods` is an ordered list of `(period_label, shap_values, feature_names)` tuples; sample counts may differ across periods.
- `concept_drift_lines(graph, df, top_k=10)` — one branch-coloured line per concept across periods. `top_k` filters to the K concepts with the largest max-across-periods value to avoid spaghetti.
- `concept_drift_delta(graph, periods, baseline=None, target=None)` — two-period delta view. Defaults baseline = first, target = last. Returns wide DataFrame with `baseline`, `target`, `delta` (= target − baseline) columns.
- `concept_drift_sunburst(graph, df)` — sunburst coloured by per-concept SHAP drift with a diverging `RdBu_r` palette centred at 0 (positive delta = red, negative = blue). Sector area uses `feature_count` (additive, safe under `branchvalues="total"`).

### Notebook

- Three new sections: **Part F** (uncertainty + interactions: F.1 bootstrap CI bar, F.2 SHAP interaction matrix with its own dedicated `TreeExplainer.shap_interaction_values` cell, F.3 multi-tier Sankey), **Part G** (cohort: G.1 segment heatmap, G.2 Pareto), **Part H** (drift: H.1 multi-period line chart, H.2 baseline → target delta sunburst).
- Synthetic cohorts (`<30 / 30-50 / 50-65 / 65+`) cut from `X_test["age"]` and synthetic periods (3 random shards with `NaN` injected into `MonthlyIncome` for `period_3`) so the demo shows visible drift on a single-snapshot dataset.

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

[Unreleased]: https://github.com/wlazlod/concept-graph-xai/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.6.0
[0.5.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.5.0
[0.4.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.4.0
[0.3.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.3.0
[0.2.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.2.0
[0.1.0]: https://github.com/wlazlod/concept-graph-xai/releases/tag/v0.1.0
