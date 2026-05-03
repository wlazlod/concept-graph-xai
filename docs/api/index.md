# API Reference

Auto-generated from source-code docstrings via [`mkdocstrings`](https://mkdocstrings.github.io/).

## Core

| Module | Description |
|--------|-------------|
| [`ConceptGraph`](graph.md) | The tree-shaped concept graph. Constructors from dict / YAML / NetworkX; deterministic DFS traversal. |
| [`Adapters`](adapters.md) | Convert SHAP, sklearn permutation results, and `model.feature_importances_` into the canonical `(values, feature_names)` tuple. |

## Metrics

Each function returns a tidy `pandas.DataFrame` indexed by concept-path.

| Module | Function(s) | Question answered |
|--------|-------------|-------------------|
| [`Counts`](metrics-counts.md) | `feature_counts` | How many features under each concept? |
| [`Importance`](metrics-importance.md) | `importance_sum` | How much importance does each concept aggregate? |
| [`Utilization`](metrics-utilization.md) | `utilization` | Which concepts does the model actually use? |
| [`Ablation`](metrics-ablation.md) | `auc_drop` (3 strategies) | How much performance is lost when a concept's data is missing? |
| [`Correlation`](metrics-correlation.md) | `feature_correlation`, `nullity_correlation`, `shap_correlation` | Are concepts internally coherent? Do they go missing together? Do features look substitutable to the model? |
| [`Missingness`](metrics-missingness.md) | `column_missing_rate`, `joint_missing_rate` | How often does a feature / a whole concept go missing? |
| [`Coherence`](metrics-coherence.md) | `coherence_importance` | Are concepts well-designed (coherent + important)? |

## Plotting

All plots return `plotly.graph_objects.Figure`. Static PNG via the `[png]` extra (`kaleido==0.2.1`).

| Module | Functions |
|--------|-----------|
| [`Plotting`](plotting.md) | `sunburst`, `utilization_map`, `auc_drop_map`, `correlation_block`, `joint_missing_map`, `coherence_importance_scatter`, `regulatory_tag_overlay` |
