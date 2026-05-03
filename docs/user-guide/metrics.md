# Metrics

Every metric returns a tidy `pandas.DataFrame` indexed by `path` (the concept's full DFS path joined with `/`). This makes them composable, easy to test, and trivially serialisable to CSV / Parquet for downstream dashboards.

## Layered overview

| Family | Functions | Question |
|---|---|---|
| **Counts** | [`feature_counts`](../api/metrics-counts.md) | How many features under each concept? |
| **Importance** | [`importance_sum`](../api/metrics-importance.md) | How much importance does each concept aggregate? |
| **Utilization** | [`utilization`](../api/metrics-utilization.md) | Which concepts does the model actually use? |
| **Ablation** | [`auc_drop`](../api/metrics-ablation.md) | How much performance drops when a concept's data is missing? |
| **Correlation** | [`feature_correlation`](../api/metrics-correlation.md), [`nullity_correlation`](../api/metrics-correlation.md), [`shap_correlation`](../api/metrics-correlation.md) | Do concepts cluster as the tree says they should? |
| **Missingness** | [`column_missing_rate`](../api/metrics-missingness.md), [`joint_missing_rate`](../api/metrics-missingness.md) | How often does a feature / a whole concept go missing? |
| **Coherence** | [`coherence_importance`](../api/metrics-coherence.md) | Are concepts well-designed (coherent + important)? |

## Standard input shape

Every metric that takes per-feature signal accepts the same canonical shape:

* `feature_names: Sequence[str]` of length F;
* `importances: np.ndarray` of shape `(F,)` (already aggregated) or `(N, F)` (per-sample SHAP-style).

`(N, F)` is collapsed by the metric layer using `np.abs(arr).mean(axis=0)` by default. Switch to signed aggregation with `signed=True` or to summed aggregation with `agg="sum"`.

## Standard output shape

Every metric returns a DataFrame with at least:

| Column | Type | Meaning |
|--------|------|---------|
| `name` | `str` | Bare node name. |
| `kind` | `"concept"` \| `"feature"` | Node kind. |
| `depth` | `int` | Distance from the root. |
| `parent` | `str` | Parent name (empty string for root). |

…plus metric-specific columns. The DataFrame index is the `/`-joined concept path (so two different `Age` nodes in two different sub-trees never collide).

## Choosing between metrics

### "How big is each branch?"
Use [`feature_counts`](../api/metrics-counts.md). Pure structural — does not need any model.

### "Which concept does the model rely on the most?"
Use [`importance_sum`](../api/metrics-importance.md) with SHAP values. Pair with the [`sunburst`](plotting.md#sunburst) plot.

### "Are there parts of my tree the model ignores?"
Use [`utilization`](../api/metrics-utilization.md), pair with [`utilization_map`](plotting.md#utilization-map).

### "How much performance is at risk if a whole branch's data goes missing?"
Use [`auc_drop`](../api/metrics-ablation.md). Pick a strategy (see [Ablation Strategies](ablation.md)).

### "Is the concept tree itself any good?"
Use the §H trio: [`feature_correlation`](../api/metrics-correlation.md) + [`coherence_importance`](../api/metrics-coherence.md) + [`coherence_importance_scatter`](plotting.md#coherence-importance-scatter). See [Concept-Design Diagnostics](diagnostics.md).

### "Do whole concepts go missing together?"
Use [`nullity_correlation`](../api/metrics-correlation.md) and [`joint_missing_rate`](../api/metrics-missingness.md), pair with [`correlation_block`](plotting.md#correlation-block) and [`joint_missing_map`](plotting.md#joint-missing-map).

### "Does the model treat two concepts as substitutes?"
Use [`shap_correlation`](../api/metrics-correlation.md), pair with [`correlation_block`](plotting.md#correlation-block).

## Composing metrics

Because every output is keyed by `path`, joining is trivial:

```python
import pandas as pd
from concept_graph_xai import (
    coherence_importance,
    column_missing_rate,
    importance_sum,
    joint_missing_rate,
    auc_drop,
)

imp = importance_sum(graph, names, shap_values).reset_index()
jmr = joint_missing_rate(graph, X_test).reset_index()
adp = auc_drop(graph, model, X_test, y_test, feature_names=names,
               strategy="permutation").reset_index()

summary = (
    imp.merge(jmr[["path", "joint_missing_rate"]], on="path")
       .merge(adp[["path", "auc_drop_mean"]], on="path")
)
```

The "realism-weighted AUC drop" view — `auc_drop_mean × joint_missing_rate` — is intentionally **not** computed by the package; it is one line in the join above.

## Defaults that bite

* `auc_drop` skips the **root** concept by default (`skip_root=True`). The root represents "everything is missing", which trivially zeros the score and is rarely useful.
* `importance_sum` uses **mean(|SHAP|)** by default. Use `signed=True` for direction-aware aggregation (concepts that consistently push the prediction up vs. down).
* `utilization` flags a feature as used when **|importance| > threshold**, default `0.0`. Raise the threshold to mask noise; lower it to be inclusive.
* `feature_correlation` and friends default to **Spearman** rank correlation — robust to monotonic non-linearities, closer to what tree models exploit.
