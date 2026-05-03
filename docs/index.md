# concept-graph-xai

**Concept-graph aware visualisation of model feature usage and importance, with concept-level ablation and design diagnostics.**

---

`concept-graph-xai` lets you supply an external **business-concept tree** that maps your raw model features (e.g. `age`, `n_30_59_dpd`, `monthly_income`) to higher-level concepts (`Demographics`, `Behaviour > Delinquency`, `Income`) — and then plots model behaviour at *the concept level*, not the feature level.

## Why concept-graph-xai?

Most interpretability tools (SHAP, Shapash, the standard `seaborn.clustermap`) either work per-feature or build a **data-driven** hierarchy by clustering correlated features. Neither is the same as the hierarchy the *business* operates on.

`concept-graph-xai` solves this by:

- accepting an **externally supplied** concept tree (YAML, dict, or NetworkX);
- aggregating any per-feature signal up the tree (counts, SHAP, importance, missingness, correlation);
- shipping interactive **Plotly** sunbursts, heatmaps, and scatters that you can also render to PNG via `kaleido`;
- separating the metric layer (`metrics/*` returns tidy DataFrames) from the plot layer (`plotting/*` consumes them) so each is independently testable.

## Quick example

```python
from concept_graph_xai import (
    ConceptGraph,
    auc_drop,
    auc_drop_map,
    feature_counts,
    importance_sum,
    sunburst,
    utilization,
    utilization_map,
)
from concept_graph_xai.adapters import from_shap_explanation

graph = ConceptGraph.from_dict({
    "Risk": {
        "Demographics": {"Age": ["age"], "Family": ["n_dependents"]},
        "Income": ["monthly_income", "debt_ratio"],
        "Behaviour": {
            "Delinquency": ["n_30_59_dpd", "n_60_89_dpd", "n_90_plus_dpd"],
            "Utilization": ["revolving_utilization"],
        },
    }
})

shap_values, names = from_shap_explanation(explanation,
                                           feature_names=X.columns.tolist())

sunburst(graph, importance_sum(graph, names, shap_values),
         value="importance_sum", colorscale="Viridis").show()

util_df = utilization(graph, names, shap_values, threshold=0.0)
utilization_map(graph, util_df).show()

drop_df = auc_drop(graph, model, X_test, y_test,
                   feature_names=X_test.columns.tolist(),
                   strategy="permutation", n_repeats=10)
auc_drop_map(graph, drop_df).show()
```

## What you can plot

| Plot | Question it answers |
|------|---------------------|
| `sunburst(graph, feature_counts(graph))` | How many features are mapped under each concept? |
| `sunburst(graph, importance_sum(...))` | How much importance does each concept carry? |
| `utilization_map(graph, utilization(...))` | Which parts of the graph does my model actually use? |
| `auc_drop_map(graph, auc_drop(...))` | How much AUC do I lose if a whole concept's data goes missing? |
| `correlation_block(feature_correlation(graph, X))` | Are the supplied concepts internally coherent? Are concept boundaries leaky? |
| `correlation_block(nullity_correlation(graph, X))` | Do features inside a concept go missing together? |
| `joint_missing_map(graph, joint_missing_rate(...))` | How often does a *whole branch* go missing in production? |
| `coherence_importance_scatter(coherence_importance(...))` | Which concepts are well-designed, kitchen sinks, redundant, or noise? |
| `correlation_block(shap_correlation(...))` | Which features does the model treat as substitutable, regardless of raw correlation? |
| `regulatory_tag_overlay(graph, tag_key="tag")` | How much of the model's decision flows through PII / financial / behavioural concepts? |

## Architecture at a glance

```
ConceptGraph (tree, NetworkX-backed)
        │
        ▼
metrics/*  →  pandas.DataFrame indexed by concept-path
        │
        ▼
plotting/* →  plotly.graph_objects.Figure (PNG via kaleido)

adapters/* →  shap.Explanation / permutation_importance / model.feature_importances_
              into the canonical (values, feature_names) tuple
```

The metric layer never imports plotly; the plot layer never touches the model. Both are independently testable.

[Get started :material-arrow-right:](getting-started.md){ .md-button .md-button--primary }
[See the API :material-arrow-right:](api/index.md){ .md-button }

## Roadmap

* **v0.1** — counts, importance, utilization, three ablation strategies, three sunburst plots. ✅
* **v0.2** — bug-fix release for `auc_drop_map`. ✅
* **v0.3 (current)** — concept-design diagnostics: block correlation matrices (feature, nullity, SHAP), joint-missing-rate sunburst, coherence-vs-importance scatter, regulatory-tag overlay. ✅
* **v0.4** — concept beeswarm, signed bar with bootstrap CIs, `ConceptPredictionExplainer.waterfall`.
* **v0.5** — SHAP-interaction heatmap (C×C), concept Sankey, segment heatmap, segment Pareto, attribution drift.
* **v0.6** — protected-group disparity heatmap.
* **v1.0** — DAG support with optional per-edge weights.

See the [full roadmap](roadmap.md) for milestone-level status.
