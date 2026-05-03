# concept-graph-xai

Concept-graph aware visualisation of model feature usage and importance, with concept-level ablation metrics.

> Status: **alpha (v0.2.0)**. API may change between minor releases.
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
| `sunburst(graph, feature_counts(...))` | How many features are mapped under each concept? |
| `sunburst(graph, importance_sum(...), colorscale="Viridis")` | How much importance does each concept carry? |
| `utilization_map(graph, utilization(...))` | Which parts of the graph does my model actually use? Unused branches are shown in grey. |
| `auc_drop_map(graph, auc_drop(..., strategy="permutation"))` | How much AUC do I lose if a whole concept's data goes missing? |
| `correlation_block(feature_correlation(graph, X))` | Are the supplied concepts internally coherent? Are concept boundaries leaky? |
| `correlation_block(nullity_correlation(graph, X))` | Do features inside a concept go missing together? |
| `joint_missing_map(graph, joint_missing_rate(graph, X))` | How often does a *whole branch* go missing in production? |
| `coherence_importance_scatter(coherence_importance(...))` | Quadrant chart: which concepts are well-designed, kitchen sinks, redundant, or noise? |
| `correlation_block(shap_correlation(graph, names, shap_values))` | Which features does the model treat as substitutable, regardless of raw correlation? |
| `regulatory_tag_overlay(graph, tag_key="tag")` | How much of the model's decision flows through PII / financial / behavioural concepts? |

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

# 3. Counts and importances
counts_df = feature_counts(graph)
imp_values, names = from_feature_importances_(model, list(X_train.columns))
imp_df = importance_sum(graph, names, imp_values)

# 4. Utilization (which concepts the model actually uses)
util_df = utilization(graph, names, imp_values, threshold=0.0)

# 5. AUC loss per concept (3 strategies; pick what you need)
drop_df = auc_drop(
    graph, model, X_test, y_test,
    feature_names=list(X_test.columns),
    strategy="permutation",
    n_repeats=10,
    random_state=42,
)

# 6. Render
fig_counts = sunburst(graph, counts_df, value="count", title="Feature counts")
fig_imp    = sunburst(graph, imp_df, value="importance_sum", colorscale="Viridis")
fig_util   = utilization_map(graph, util_df)
fig_drop   = auc_drop_map(graph, drop_df)

fig_drop.write_image("auc_drop.png", scale=2)  # needs the [png] extra
```

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

See [SCOPING.md](SCOPING.md) for the full design notes.

* **v0.1**: counts, importance, utilization, three ablation strategies, three sunburst plots. ✅
* **v0.2**: bug-fix release for `auc_drop_map`. ✅
* **v0.3 (current)**: concept-design diagnostics — block correlation matrices (feature, nullity, SHAP), joint-missing-rate sunburst, coherence-vs-importance scatter, regulatory-tag overlay. ✅
* **v0.4**: concept beeswarm, signed bar with bootstrap CIs, `ConceptPredictionExplainer.waterfall`.
* **v0.5**: SHAP-interaction heatmap (C×C), concept Sankey, segment heatmap, segment Pareto, attribution drift line chart, drift delta sunburst.
* **v0.6**: protected-group disparity heatmap.
* **v1.0**: DAG support (multi-parent concepts) with optional per-edge weights and Sankey rendering.

See [PROPOSALS.md](PROPOSALS.md) for the locked decisions and per-proposal status.

## Development

```bash
uv sync --all-extras --dev
uv run pytest
uv run ruff check .
uv run mypy
```

## License

MIT — see [LICENSE](LICENSE).
