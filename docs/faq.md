# FAQ

**Why does a metric warn about "features not in graph"?**
Your `feature_names` contains columns the graph does not cover. Every importance-driven
metric routes through the same alignment step, controlled by `on_unknown`: `"warn"`
(the default) proceeds on the covered subset, `"raise"` turns the mismatch into a
`KeyError`, `"ignore"` silences it. Use `"raise"` while developing — a typo in one
feature name silently shrinks every concept above it.

**Do I need `shap` installed to use the library?**
No. `concept_graph_xai` never imports `shap`; `from_shap_explanation` duck-types any
object with `.values` and `.feature_names` (or a raw array plus explicit
`feature_names`). The `[shap]` extra exists only as a convenience for *producing* SHAP
values in the same environment.

**Why does an adapter raise "contains NaN or Inf"?**
All three adapters (`from_shap_explanation`, `from_permutation_importance`,
`from_feature_importances_`) reject non-finite values at the boundary instead of
letting them propagate silently through every metric. A NaN here usually means the
explainer ran on rows with missing data it could not handle, or a zero-variance
feature — fix the upstream run rather than masking the values.

**Why does `fig.write_image(...)` fail when the figures render fine?**
Static PNG export goes through `kaleido`, which is deliberately not a core dependency —
install the `[png]` extra (`kaleido>=0.2.1,<1`). Interactive rendering (`.show()`,
notebooks, HTML) never needs it.

**Why does `auc_drop` raise `ValueError` about missing arguments?**
Two strategies need extra inputs: `strategy="retrain"` requires `train_fn=` (a
callable rebuilding your model from scratch) plus `X_train` and `y_train`;
`strategy="shap_marginal"` requires `shap_values=` (shape `(N, F)`) and
`base_predictions=` (shape `(N,)`). Only the default `strategy="permutation"` works
from the fitted model alone.

**Is `auc_drop` limited to AUC?**
No — the name is historical. The `metric` argument accepts any sklearn scorer name
(resolved via `get_scorer`) or any `callable(y, predictions) -> float`; the default is
`"roc_auc"`.

**Why is the root's `auc_drop_mean` NaN, and why is the root missing from sunbursts?**
Ablating the root means "the model gets pure noise", which trivially tanks the score
and would dominate the colour scale, so `skip_root=True` (the default) leaves the root
row in the DataFrame with a NaN drop. Structural columns like `feature_count` are still
populated for every node including the root — Plotly's `branchvalues="total"` silently
drops the whole chart when a parent's value is smaller than the sum of its children.
Sunburst plots hide the root sector by default for the same legibility reason
(`hide_root=False` keeps it).

**Why does `ConceptGraph` construction raise `ValueError`?**
The tree invariants are enforced eagerly: exactly one root, every leaf is a `feature`,
every internal node is a `concept` with at least one descendant feature, node names
unique across the whole graph, no cycles or multiple parents. The message names the
offending node (e.g. `Duplicate node name: 'age'`, `concept 'X' has no children
(orphan concept)`). DAGs are reserved for v1.0 — see the [roadmap](roadmap.md).

**Why does `concept_disparity` require `reference=`?**
The protected-attribute API is deliberately minimal in v0.6: one attribute per call,
and every group is compared against one explicit baseline group rather than all
pairwise combinations. The reference group's row is kept in the output with `value=0`
so the heatmap shows it as the visible zero baseline. Intersectional attributes and
population-mean baselines are filed for later — see the [roadmap](roadmap.md).
