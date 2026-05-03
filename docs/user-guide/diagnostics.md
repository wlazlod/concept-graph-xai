# Concept-Design Diagnostics

The headline v0.3 capability. **Is the concept tree itself any good?**

When someone hands you a tree, the obvious question is whether the concepts it defines actually correspond to *clusters* in the feature data and *groups* in the model's behaviour. The §H trio answers that.

## The three views

| View | Operates on | Diagnoses |
|---|---|---|
| `feature_correlation` + `correlation_block` | Raw `X` | Are the features inside a concept correlated? Are concept boundaries leaky? |
| `nullity_correlation` + `correlation_block` | `X.isna()` | Do features inside a concept go missing together? |
| `shap_correlation` + `correlation_block` | `shap_values` | Does the model treat features inside a concept as substitutes? |

All three return a `CorrelationResult` that the same [`correlation_block`](plotting.md#correlation-block) plot can render.

## Reading the block heatmap

Features are reordered along the **graph's DFS preorder**, so every concept's descendants form a contiguous block. The plot draws separator lines between top-level concept blocks and annotates each diagonal block with its mean(|ρ|).

A "good" tree typically looks like this when the matrix is **feature correlation**:

* dark, high-correlation diagonal blocks (within-concept coherence);
* mostly white off-diagonal blocks (concepts are different things).

Boundary leakage shows up as a high off-diagonal block — two concepts that turned out to be more correlated with each other than with their internal features. That's a tree-design problem, not a model problem.

## The headline scatter

The single most actionable plot. One point per concept; x = within-concept mean(|ρ|), y = summed |SHAP|.

```python
from concept_graph_xai import (
    coherence_importance,
    coherence_importance_scatter,
)

coh_df = coherence_importance(graph, X_test, feature_names, shap_values)
coherence_importance_scatter(coh_df).show()
```

| Quadrant | What to do |
|---|---|
| **Well-designed** (high coherence, high importance) | Keep. Document. |
| **Kitchen sink** (low coherence, high importance) | Split. The model relies on this concept but the concept itself is heterogeneous — there are probably two or three sub-concepts hidden inside. |
| **Redundant** (high coherence, low importance) | Merge or drop. The features within are coherent (they're really one concept), but the model doesn't lean on them. |
| **Noise** (low on both) | Drop. Adds bookkeeping cost without explanatory value. |

The default thresholds are the medians across concepts; pass `coherence_threshold=` and `importance_threshold=` to use absolute cutoffs.

## When `feature_correlation` and `shap_correlation` disagree

This is informative, not a bug. Two raw-uncorrelated features can be **SHAP-redundant** — the model treats them as substitutes — when:

* one is a transformation of the other that the model approximates internally;
* they encode the same upstream signal through different representations (e.g. `monthly_income` and `log(monthly_income)`).

Conversely, two raw-correlated features can be **SHAP-different** — the model uses them in opposite directions in different parts of the input space.

## Missingness-side diagnostics

`feature_correlation` and `shap_correlation` answer "is the tree shape right for this data and this model". `nullity_correlation` answers "if a concept's data goes missing, does it tend to do so as a whole block?".

```python
from concept_graph_xai import (
    nullity_correlation,
    joint_missing_rate,
    correlation_block,
    joint_missing_map,
)

correlation_block(nullity_correlation(graph, X_test),
                  title="Do features go missing together?").show()

joint_missing_map(graph, joint_missing_rate(graph, X_test),
                  title="Whole-branch missingness").show()
```

If the diagonal blocks are dark, the "whole branch missing" scenario simulated by [`auc_drop`](../api/metrics-ablation.md) is realistic, and the AUC-drop view should be taken seriously. If they're light, the simulation is overly pessimistic — single-feature outages are far more common than block outages.

## A complete diagnostic walkthrough

```python
from concept_graph_xai import (
    coherence_importance,
    coherence_importance_scatter,
    correlation_block,
    feature_correlation,
    joint_missing_map,
    joint_missing_rate,
    nullity_correlation,
    shap_correlation,
)

# 1. Are the concepts coherent?
fc = feature_correlation(graph, X_test, method="spearman")
correlation_block(fc, title="Feature correlation").show()
fc.block_stats.sort_values("mean_abs", ascending=False)

# 2. Does the model treat them as such?
sc = shap_correlation(graph, feature_names, shap_values)
correlation_block(sc, title="SHAP correlation").show()

# 3. Are concepts well-designed (the headline)?
coh_df = coherence_importance(graph, X_test, feature_names, shap_values)
coherence_importance_scatter(coh_df).show()

# 4. Do whole branches go missing together?
correlation_block(nullity_correlation(graph, X_test),
                  title="Nullity correlation").show()
joint_missing_map(graph, joint_missing_rate(graph, X_test)).show()
```

Run this once after building the tree. Any "kitchen sink" concept is a signal to split before going further.

## What this is *not*

* Not a way to **derive** a tree from the data. There are good clustering tools for that (`shap.utils.hclust`, `seaborn.clustermap`); using them after building a business-driven tree just produces a different tree, not feedback on the one you wrote.
* Not a substitute for a domain-expert review. The diagnostic flags concepts as "kitchen sink" or "redundant" *given the data and model*. Many regulatory or business reasons to keep a concept anyway.
