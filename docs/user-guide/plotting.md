# Plotting

Every plot is a `plotly.graph_objects.Figure`. They render in notebooks, on dashboards, and to static PNG via `kaleido` (the `[png]` extra). The plot layer **never** imports the model — it consumes a `ConceptGraph` and a metric DataFrame.

## Sunburst family

These three share the same radial layout (`go.Sunburst`). They differ only in what the segments mean.

### `sunburst` { #sunburst }

Generic concept sunburst. Pass the metric DataFrame and the column you want the segment area to encode.

```python
from concept_graph_xai import sunburst, feature_counts, importance_sum

sunburst(graph, feature_counts(graph), value="count").show()

sunburst(graph,
         importance_sum(graph, names, shap_values),
         value="importance_sum",
         colorscale="Viridis").show()
```

Key parameters:

| Parameter | Effect |
|---|---|
| `value` | Column used for **segment area**. |
| `colorscale` | Plotly colorscale. When set, segments are colored by `color_value` (defaults to `value`). |
| `branchvalues` | `"total"` (default) or `"remainder"`. Controls how Plotly interprets parent / child values. |
| `extra_hover` | Extra DataFrame columns to append to the hover tooltip. |

### `utilization_map` { #utilization-map }

Variant of `sunburst` where unused branches are rendered grey.

```python
from concept_graph_xai import utilization, utilization_map

util_df = utilization(graph, names, shap_values, threshold=0.0)
utilization_map(graph, util_df,
                used_color="#1f77b4",
                unused_color="#d3d3d3").show()
```

Used / unused is determined by the `is_used` column the [`utilization`](../api/metrics-utilization.md) metric writes.

### `auc_drop_map` { #auc-drop-map }

Variant where each segment is **coloured by AUC drop**.

```python
from concept_graph_xai import auc_drop, auc_drop_map

drop_df = auc_drop(graph, model, X_test, y_test,
                   feature_names=X_test.columns.tolist(),
                   strategy="permutation")
auc_drop_map(graph, drop_df, colorscale="Reds").show()
```

The colorscale is automatically diverging when both positive and negative drops exist (an "improvement" under permutation usually means a feature was injecting noise).

### `joint_missing_map` { #joint-missing-map }

Variant where each segment is coloured by **joint-missing rate** (P15b / P13).

```python
from concept_graph_xai import joint_missing_rate, joint_missing_map

jmr_df = joint_missing_rate(graph, X_test)
joint_missing_map(graph, jmr_df).show()
```

### `regulatory_tag_overlay` { #regulatory-tag-overlay }

Categorical sunburst — segments coloured by a metadata tag stored on each `ConceptGraph` node.

```python
from concept_graph_xai import regulatory_tag_overlay

regulatory_tag_overlay(graph, tag_key="tag").show()
```

The tag is read from `graph.view(node).metadata[tag_key]`. Set it by either:

- attaching `metadata={"tag": "PII"}` when building from NetworkX, or
- editing `graph.graph.nodes[name]["metadata"]` directly after construction.

## Correlation family

These three share the same `go.Heatmap` layout but differ in *what* matrix they show.

### `correlation_block` { #correlation-block }

Renders any [`CorrelationResult`](../api/metrics-correlation.md): block-structured heatmap with concept separators and `mean(|ρ|)` annotations on each diagonal block.

```python
from concept_graph_xai import (
    feature_correlation, nullity_correlation, shap_correlation,
    correlation_block,
)

correlation_block(feature_correlation(graph, X_test)).show()
correlation_block(nullity_correlation(graph, X_test)).show()
correlation_block(shap_correlation(graph, names, shap_values)).show()
```

Diagonal blocks reveal **within-concept coherence**; off-diagonal blocks reveal **boundary leakage**.

## Diagnostic family

### `coherence_importance_scatter` { #coherence-importance-scatter }

The headline v0.3 diagnostic. Quadrant scatter: x = within-concept mean(|ρ|), y = summed |SHAP|.

```python
from concept_graph_xai import coherence_importance, coherence_importance_scatter

coh_df = coherence_importance(graph, X_test, names, shap_values)
coherence_importance_scatter(coh_df).show()
```

Quadrants:

| Quadrant | Reading |
|---|---|
| **green** — well-designed | High coherence + high importance. Keep. |
| **red** — kitchen sink | Low coherence + high importance. Split this concept. |
| **orange** — redundant | High coherence + low importance. Could be merged or dropped. |
| **grey** — noise | Low on both axes. Definitely drop. |

The dashed reference lines come from the median across concepts by default; pass explicit `coherence_threshold` and `importance_threshold` arguments to override.

## Static PNG export

Every figure can be rendered to PNG via `kaleido`:

```python
fig = correlation_block(feature_correlation(graph, X_test))
fig.write_image("feature_correlation.png", width=900, height=900, scale=2)
```

The `[png]` extra pins `kaleido==0.2.1` (older but Chrome-free; v1 requires Chrome at runtime).

## Layout customisation

Every plot accepts a `layout_kwargs: dict[str, Any]` parameter passed verbatim to `fig.update_layout`. Use it for figure size, font, legend placement, etc.:

```python
sunburst(graph, counts_df, value="count",
         layout_kwargs={"width": 800, "height": 800,
                        "font": {"family": "Inter"}})
```

## Anti-patterns

* **Don't pass an unrelated DataFrame to a sunburst.** All sunburst plots reindex by concept-path; if your DataFrame isn't keyed that way they will silently render empty rings. Use the metric functions to produce inputs.
* **Don't plot the matrix with `branchvalues="total"` when a parent's value is smaller than the sum of its children's.** Plotly silently drops the chart. (This was a real v0.1 bug; v0.2 fixes the only path where the package itself produced such inputs.)
* **Don't try to cast a sunburst to a treemap by passing `kind="treemap"`.** Use the dedicated function — there isn't one yet, but it's on the v0.4 backlog.
