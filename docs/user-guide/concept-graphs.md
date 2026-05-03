# Concept Graphs

A `ConceptGraph` is a **tree** of business concepts whose leaves are model features.

## The data model

In v1 every node is either a `concept` (internal) or a `feature` (leaf):

* the tree has **exactly one root** and the root is a concept;
* every leaf is a feature, every concept has at least one descendant feature;
* node names are unique across the whole graph;
* every node may carry an arbitrary `metadata` dict (used by the [regulatory tag overlay](plotting.md#regulatory-tag-overlay)).

These invariants are enforced at construction time. DAGs (a feature mapped to multiple parents) are reserved for v1.0 — see the [roadmap](../roadmap.md).

## Building a graph

### From a nested dict

The most common form. **Dicts are concepts, lists are feature leaves**:

```python
from concept_graph_xai import ConceptGraph

graph = ConceptGraph.from_dict({
    "RiskProfile": {
        "Demographics": {
            "Age": ["age"],
            "Family": ["n_dependents"],
        },
        "Income": ["monthly_income", "debt_ratio"],
        "Behaviour": {
            "Delinquency": ["n_30_59_dpd", "n_60_89_dpd", "n_90_plus_dpd"],
            "Utilization": ["revolving_utilization"],
        },
    }
})
```

The single top-level key (`RiskProfile`) becomes the root.

### From YAML

The same shape as the nested dict, just on disk:

```yaml
# concepts.yaml
RiskProfile:
  Demographics:
    Age: [age]
    Family: [n_dependents]
  Income: [monthly_income, debt_ratio]
  Behaviour:
    Delinquency: [n_30_59_dpd, n_60_89_dpd, n_90_plus_dpd]
    Utilization: [revolving_utilization]
```

```python
graph = ConceptGraph.from_yaml("concepts.yaml")
```

### From NetworkX

If you already have a `networkx.DiGraph` with `kind` and `metadata` attributes set per node, you can use it directly. This is the path you take when the tree is generated from another tool (e.g. an ontology library) and when you want to attach metadata for the [regulatory tag overlay](plotting.md#regulatory-tag-overlay):

```python
import networkx as nx
from concept_graph_xai import ConceptGraph

g = nx.DiGraph()
g.add_node("Risk", kind="concept", metadata={})
g.add_node("Demographics", kind="concept", metadata={"tag": "PII"})
g.add_node("age", kind="feature", metadata={"tag": "PII"})
g.add_edge("Risk", "Demographics")
g.add_edge("Demographics", "age")

graph = ConceptGraph.from_networkx(g, root="Risk")
```

## Node metadata

Every node has a `metadata: dict[str, Any]` slot. The package uses one well-known key:

| Key | Used by | Meaning |
|-----|---------|---------|
| `tag` | [`regulatory_tag_overlay`](plotting.md#regulatory-tag-overlay) | A categorical label such as `"PII"`, `"financial"`, `"behavioural"`. |

You can store anything else (regulatory tags, owner names, descriptions) — it travels with the graph but does not affect any metric.

## Inspecting the graph

```python
graph.root            # "RiskProfile"
graph.features()      # ['age', 'n_dependents', 'monthly_income', ...]
graph.concepts()      # ['RiskProfile', 'Demographics', 'Age', ...]
graph.parent_of("monthly_income")           # 'Income'
graph.descendant_features("Behaviour")      # ['n_30_59_dpd', ...]
graph.path("monthly_income")                # ('RiskProfile', 'Income', 'monthly_income')
graph.kind("Income")                        # 'concept'
graph.view("Demographics").metadata         # {}  (or whatever you set)
```

The traversal order is **deterministic depth-first preorder**, set at construction time. Every metric and plot uses this order so you get reproducible block layouts in the correlation matrices.

## Validation errors

`ConceptGraph` raises early. Common failures:

| Error | Cause |
|---|---|
| `ValueError: Duplicate node name` | The same name appears twice in the tree. |
| `ValueError: ConceptGraph must be a tree` | The supplied DiGraph has a cycle or multiple parents. |
| `ValueError: feature 'X' must be a leaf (has children)` | A `kind="feature"` node has outgoing edges. |
| `ValueError: concept 'X' has no children (orphan concept)` | A non-root concept node has no descendants. |

## Mismatches with the model's feature set

Two cases come up in practice:

- **Feature in the model but not in the graph.** Most metrics (`importance_sum`, `utilization`, `auc_drop`) accept an `on_unknown` parameter; the default is `"warn"`. Set it to `"raise"` if you want to force the graph to cover everything, or `"ignore"` if you accept a partial view.
- **Feature in the graph but not in the model.** Counted in `feature_counts` but not in the importance-driven metrics. The utilization map will show such features as "not used", which is correct.

## Where the graph fits

```
ConceptGraph (tree, NetworkX-backed)
        │
        ▼
metrics/*  →  pandas.DataFrame indexed by concept-path string
        │
        ▼
plotting/* →  plotly.graph_objects.Figure (PNG via kaleido)
```

The graph is the only object that crosses the metric / plot boundary. Everything else is a tidy DataFrame.
