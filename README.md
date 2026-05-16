# concept-graph-xai

Concept-graph aware visualisation of model feature usage and importance, with concept-level ablation metrics.

> Status: **alpha (v0.5.0)**. API may change between minor releases.
>
> 📖 **Docs:** <https://wlazlod.github.io/concept-graph-xai/>

## What it does

You give it:

1. a **business-concept tree** that maps your raw model features to higher-level concepts (e.g. `age, n_dependents -> Demographics`);
2. a fitted **tree model** (LightGBM / XGBoost / CatBoost / sklearn) and its **per-feature importances** (or per-sample SHAP values);
3. a held-out test set and a target.

It gives you:

| Plot | Question it answers |
|---|---|
| `sunburst(graph, importance_sum(...))` | How much importance does each concept carry? (hierarchical colour per top-level branch — sub-concepts are lighter shades) |
| `utilization_map(graph, utilization(...))` | How many features are under each concept *and* which branches does my model actually use? Sector area = feature count; used branches are coloured by family with hierarchical shading; unused branches are grey. |
| `auc_drop_map(graph, auc_drop(..., strategy="permutation"))` | How much AUC do I lose if a whole concept's data goes missing? |
| `correlation_block(feature_correlation(graph, X))` | Are the supplied concepts internally coherent? Are concept boundaries leaky? |
| `correlation_block(nullity_correlation(graph, X))` | Do features inside a concept go missing together? |
| `joint_missing_map(graph, joint_missing_rate(graph, X))` | How often does a *whole branch* go missing in production? |
| `coherence_importance_scatter(coherence_importance(...))` | Quadrant chart: which concepts are well-designed, kitchen sinks, redundant, or noise? |
| `correlation_block(shap_correlation(graph, names, shap_values))` | Which features does the model treat as substitutable, regardless of raw correlation? |
| `regulatory_tag_overlay(graph, tag_key="tag")` | How much of the model's decision flows through PII / financial / behavioural concepts? |
| `signed_concept_bar(graph, bootstrap_importance(...))` | What's the direction of each concept's SHAP, with bootstrap confidence intervals? |
| `concept_interaction_heatmap(concept_interaction_matrix(...))` | Which concept pairs interact (using SHAP interaction values)? |
| `concept_sankey(graph, feature_names, shap_values)` | How does SHAP flow from features through every concept tier to +/− outcomes? |
| `segment_concept_heatmap(graph, segment_importance(...))` | Which concepts matter more in which cohort? |
| `concept_pareto(graph, segment_importance(...))` | How concentrated is each cohort's SHAP across concepts? |
| `concept_drift_lines(graph, attribution_drift(...))` | How does concept importance shift across periods? |
| `concept_drift_sunburst(graph, concept_drift_delta(...))` | Which concepts changed the most between a baseline and a target period? |

The metric layer (`concept_graph_xai.metrics.*`) returns plain `pandas.DataFrame`s and never imports plotly. The plot layer takes those DataFrames and a `ConceptGraph` and returns `plotly.graph_objects.Figure`s, exportable to PNG via `kaleido`.

## Install

```bash
uv add concept-graph-xai          # core only
uv add 'concept-graph-xai[shap]'  # + SHAP adapter
uv add 'concept-graph-xai[png]'   # + kaleido for static PNG export
```

For the example notebook:

```bash
uv add 'concept-graph-xai[notebook]'
```

## Quickstart

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

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
from concept_graph_xai.adapters import from_feature_importances_

# 1. Define the tree (concepts as dicts, features as leaves)
graph = ConceptGraph.from_dict({
    "Risk": {
        "Demographics": {"Age": ["age"], "Family": ["n_dependents"]},
        "Income": ["monthly_income", "debt_ratio"],
        "Behaviour": {
            "Delinquency": ["n_30_59_dpd", "n_60_89_dpd", "n_90_plus_dpd"],
            "Utilization": ["revolving_utilization"],
        },
        "Unused": ["noise_a", "noise_b"],
    }
})

# 2. Train any tabular model (or load one)
# X is a DataFrame whose columns include every feature in the graph
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = GradientBoostingClassifier().fit(X_train, y_train)

# 3. Importances
imp_values, names = from_feature_importances_(model, list(X_train.columns))
imp_df = importance_sum(graph, names, imp_values)

# 4. Utilization (sector area = feature_count, used branches in branch hue,
#    unused branches grey — subsumes the standalone feature-counts view)
util_df = utilization(graph, names, imp_values, threshold=0.0)

# 5. AUC loss per concept (3 strategies; pick what you need)
drop_df = auc_drop(
    graph, model, X_test, y_test,
    feature_names=list(X_test.columns),
    strategy="permutation",
    n_repeats=10,
    random_state=42,
)

# 6. Render — root concept is hidden by default; pass hide_root=False to keep it.
fig_imp    = sunburst(graph, imp_df, value="importance_sum")  # branch-coloured
fig_util   = utilization_map(graph, util_df)                  # counts + utilization
fig_drop   = auc_drop_map(graph, drop_df)

fig_drop.write_image("auc_drop.png", scale=2)  # needs the [png] extra
```

`sunburst()` colours by top-level branch when no `colorscale` is given. Pass
`colorscale="Viridis"` (or `color_by="value"`) to fall back to a continuous
colorscale on `value`.

## Three ablation strategies

```python
# Cheap, model-agnostic, no retraining (default)
auc_drop(..., strategy="permutation", n_repeats=10)

# Cheapest. Approximation under SHAP additivity.
auc_drop(..., strategy="shap_marginal",
         shap_values=shap_values, base_predictions=p_baseline)

# Most faithful to "lack of data". Retrains once per concept.
auc_drop(..., strategy="retrain",
         train_fn=lambda X, y: MyModel(...).fit(X, y),
         X_train=X_train, y_train=y_train)
```

## Architecture

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

The metric layer never imports plotly, and the plot layer never touches the model. Both are independently testable.

## Roadmap

* **v0.1**: counts, importance, utilization, three ablation strategies, three sunburst plots. ✅
* **v0.2**: bug-fix release for `auc_drop_map`. ✅
* **v0.3**: concept-design diagnostics — block correlation matrices (feature, nullity, SHAP), joint-missing-rate sunburst, coherence-vs-importance scatter, regulatory-tag overlay. ✅
* **v0.4**: local explanations (`concept_violin`, `ConceptPredictionExplainer.waterfall`); rendering-default cleanups (root concept hidden by default; `sunburst` colours by branch by default; `utilization_map` subsumes the standalone feature-count sunburst with branch-hierarchical shading). ✅
* **v0.5 (current)**: uncertainty (`bootstrap_importance` + `signed_concept_bar`), interactions (`concept_interaction_matrix` + heatmap, `concept_sankey`), cohort (`segment_importance` + heatmap, `concept_pareto`), drift (`attribution_drift` + `concept_drift_lines`, `concept_drift_delta` + `concept_drift_sunburst`). ✅
* **v0.6**: protected-group disparity heatmap.
* **v1.0**: DAG support (multi-parent concepts) with optional per-edge weights and Sankey rendering.

See the [roadmap page](https://wlazlod.github.io/concept-graph-xai/roadmap/) for milestone status and the locked cross-cutting decisions.

## Development

```bash
uv sync --all-extras --dev
uv run pytest
uv run ruff check .
uv run mypy
```

## License

MIT — see [LICENSE](LICENSE).
