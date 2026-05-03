# Ablation Strategies

`auc_drop` answers a simple question — *"how much performance do I lose if a whole concept's data goes missing?"* — with three different definitions of "missing", trading off accuracy for cost.

## The three strategies

### `permutation` (default)

For each concept, **shuffle** the values of its features across rows and re-score on the held-out set, repeated `n_repeats` times.

* Cheap — no retraining.
* Model-agnostic — works for any model with `predict_proba` / `decision_function`.
* The model still *expects* the features to be there; it just sees decorrelated values. This is an upper bound on "what would happen if the data became uninformative", **not** "what would happen if the column was deleted".

```python
auc_drop(graph, model, X_test, y_test,
         feature_names=X_test.columns.tolist(),
         strategy="permutation",
         n_repeats=10, random_state=42)
```

Output columns: `auc_drop_mean`, `auc_drop_std` (across repeats), `ablated_score_mean`, `baseline_score`, `feature_count`, `strategy`.

### `retrain`

For each concept, **drop** its features from the training set, retrain, score on the held-out set with the same columns dropped.

* Most faithful to "this column will not exist in production".
* Most expensive: one retrain per concept.
* Requires a user-supplied `train_fn(X_train, y_train) -> fitted_estimator` callable so the package can rebuild the model exactly the way you want.

```python
def train_lgb(X, y):
    m = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
    m.fit(X, y)
    return m

auc_drop(graph, model, X_test, y_test,
         feature_names=X_test.columns.tolist(),
         strategy="retrain",
         train_fn=train_lgb,
         X_train=X_train, y_train=y_train)
```

Output columns same as permutation, but `auc_drop_std` is `NaN` (single retrain, no spread).

### `shap_marginal`

Cheapest. **Subtract** the concept's SHAP contributions from the prediction logits, apply the link function, re-score.

* Almost free once you have SHAP values.
* An approximation under SHAP additivity. Treats each concept's contribution as independently subtractable, which is exact only for additive models.
* Useful as a sanity check or for fast iteration; not for regulatory submissions.

```python
auc_drop(graph, model, X_test, y_test,
         feature_names=X_test.columns.tolist(),
         strategy="shap_marginal",
         shap_values=shap_values,
         base_predictions=p_test)
```

`base_predictions` should be the model's predicted probabilities on `X_test` (1D, length N).

## Picking a strategy

| Use case | Strategy |
|---|---|
| Quick iteration, "is this branch important?" | `permutation` |
| Sanity check, you already have SHAP values | `shap_marginal` |
| Regulatory submission, "what happens if Equifax goes down?" | `retrain` |
| Comparison study (run all three, look for disagreements) | All three with the same graph |

A common pattern: run `permutation` first to identify the top-k concepts, then `retrain` only on those k.

## Custom scoring metrics

`auc_drop` accepts any sklearn-compatible scoring callable via the `metric` argument.

```python
from sklearn.metrics import log_loss

# Built-in scorer name
auc_drop(..., metric="roc_auc")

# Any callable (y_true, y_score) -> float
auc_drop(..., metric=lambda y, p: -log_loss(y, p))
```

The function name (`auc_drop`) is historical — the implementation is metric-agnostic. v0.5 will add a `metric_drop` alias and deprecate the AUC-only naming.

## What `skip_root` does

The root concept covers *every* feature; ablating it means "the model gets pure noise / nothing", which by definition tanks the score.

`skip_root=True` (the default) leaves the root row in the DataFrame with `auc_drop_mean = NaN` so it does not pollute the colour scale of [`auc_drop_map`](plotting.md#auc-drop-map). The structural columns (`feature_count`, etc.) are still populated — required so the parent sunburst sector isn't smaller than its children.

## Realism — does this scenario actually happen?

`auc_drop` tells you *what would happen* if a concept went missing. To answer *how often that actually happens*, pair it with [`joint_missing_rate`](../api/metrics-missingness.md):

```python
import pandas as pd
from concept_graph_xai import auc_drop, joint_missing_rate

drop = auc_drop(graph, model, X_test, y_test,
                feature_names=X_test.columns.tolist(),
                strategy="permutation").reset_index()
jmr  = joint_missing_rate(graph, X_test).reset_index()

risk = (drop[["path", "auc_drop_mean"]]
        .merge(jmr[["path", "joint_missing_rate"]], on="path")
        .assign(realism_weighted=lambda d:
                d["auc_drop_mean"] * d["joint_missing_rate"]))
```

The package deliberately does **not** bake `realism_weighted` into `auc_drop` (decision **D2** in [`PROPOSALS.md`](https://github.com/wlazlod/concept-graph-xai/blob/main/PROPOSALS.md)) — different teams want different fusion rules, and the join is one line.
