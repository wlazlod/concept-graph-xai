# Getting Started

## Requirements

- Python ≥ 3.12
- A trained model with per-feature importance — typically a tree ensemble (LightGBM, XGBoost, CatBoost, scikit-learn `GradientBoosting*`)
- Optional but recommended: [`shap`](https://shap.readthedocs.io/) for per-sample SHAP values

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/wlazlod/concept-graph-xai.git
cd concept-graph-xai
uv sync --all-extras --dev
```

### From PyPI

```bash
# Core only — numpy, pandas, networkx, plotly, scikit-learn, pyyaml
uv add concept-graph-xai

# With the SHAP adapter
uv add 'concept-graph-xai[shap]'

# With static PNG export via kaleido
uv add 'concept-graph-xai[png]'

# Everything you need for the example notebook
uv add 'concept-graph-xai[notebook]'
```

## Quick Start

### 1. Train a tabular model

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
```

### 2. Define the concept graph

The graph is a tree: dicts are concepts, lists are feature leaves.

```python
from concept_graph_xai import ConceptGraph

graph = ConceptGraph.from_dict({
    "RiskProfile": {
        "Demographics": {"Age": ["age"], "Family": ["NumberOfDependents"]},
        "Income": ["MonthlyIncome", "DebtRatio"],
        "Behaviour": {
            "Delinquency": [
                "NumberOfTime30-59DaysPastDueNotWorse",
                "NumberOfTime60-89DaysPastDueNotWorse",
                "NumberOfTimes90DaysLate",
            ],
            "Utilization": ["RevolvingUtilizationOfUnsecuredLines"],
        },
    }
})
```

You can also load from YAML:

```python
graph = ConceptGraph.from_yaml("concepts.yaml")
```

### 3. Get importances into the canonical shape

`concept-graph-xai` accepts plain `(values, feature_names)` arrays. Use an adapter when convenient.

=== "From SHAP"

    ```python
    import shap
    from concept_graph_xai.adapters import from_shap_explanation

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)
    shap_values, feature_names = from_shap_explanation(
        explanation, feature_names=X_test.columns.tolist()
    )
    ```

=== "From permutation_importance"

    ```python
    from sklearn.inspection import permutation_importance
    from concept_graph_xai.adapters import from_permutation_importance

    result = permutation_importance(model, X_test, y_test, n_repeats=10)
    importances, feature_names = from_permutation_importance(
        result, X_test.columns.tolist()
    )
    ```

=== "From `model.feature_importances_`"

    ```python
    from concept_graph_xai.adapters import from_feature_importances_

    importances, feature_names = from_feature_importances_(
        model, X_test.columns.tolist()
    )
    ```

### 4. Compute and plot

```python
from concept_graph_xai import (
    feature_counts, importance_sum, utilization,
    sunburst, utilization_map,
)

counts_df = feature_counts(graph)
imp_df    = importance_sum(graph, feature_names, shap_values)
util_df   = utilization(graph, feature_names, shap_values, threshold=0.0)

sunburst(graph, counts_df, value="count", title="Features per concept").show()
sunburst(graph, imp_df, value="importance_sum",
         colorscale="Viridis",
         title="Concept importance (mean |SHAP|)").show()
utilization_map(graph, util_df).show()
```

### 5. Concept-level AUC drop

```python
from concept_graph_xai import auc_drop, auc_drop_map

drop_df = auc_drop(
    graph, model, X_test, y_test,
    feature_names=X_test.columns.tolist(),
    strategy="permutation",     # | "retrain" | "shap_marginal"
    n_repeats=10, random_state=42,
)
auc_drop_map(graph, drop_df).show()
```

The three strategies are explained in [Ablation Strategies](user-guide/ablation.md).

### 6. Concept-design diagnostics (v0.3)

Once you have a tree and a fitted model, ask: **is the tree any good?**

```python
from concept_graph_xai import (
    feature_correlation, nullity_correlation, shap_correlation,
    coherence_importance, joint_missing_rate,
    correlation_block, joint_missing_map, coherence_importance_scatter,
)

# Are the concepts internally coherent? Are boundaries leaky?
fc = feature_correlation(graph, X_test)
correlation_block(fc, title="Feature correlation").show()

# Do whole branches go missing together?
nc = nullity_correlation(graph, X_test)
correlation_block(nc, title="Nullity correlation").show()

joint_missing_map(graph, joint_missing_rate(graph, X_test)).show()

# The headline diagnostic: which concepts are well-designed?
coh_df = coherence_importance(graph, X_test, feature_names, shap_values)
coherence_importance_scatter(coh_df).show()
```

See [Concept-Design Diagnostics](user-guide/diagnostics.md) for the full workflow.

## Static PNG export

```python
fig = sunburst(graph, counts_df)
fig.write_image("counts.png", width=900, height=900, scale=2)
```

Requires the `[png]` extra (`kaleido==0.2.1` to avoid Chrome dependency).

## What's next?

- Read [Concept Graphs](user-guide/concept-graphs.md) to understand the graph model.
- See all [metrics](user-guide/metrics.md) and [plots](user-guide/plotting.md).
- Browse the [API Reference](api/index.md).
- Walk through the example notebook in [`examples/01_credit_risk_walkthrough.ipynb`](https://github.com/wlazlod/concept-graph-xai/blob/main/examples/01_credit_risk_walkthrough.ipynb) — runs end-to-end on the Give Me Some Credit Kaggle dataset.
