# Getting Started

The shortest path from install to first sunburst. Once you have it
working, the [Tour](tour.md) walks the full credit-risk scenario.

## Requirements

- Python ≥ 3.12
- A trained model with per-feature importance — typically a tree
  ensemble (LightGBM, XGBoost, CatBoost, scikit-learn
  `GradientBoosting*`)
- Optional but recommended: [`shap`](https://shap.readthedocs.io/) for
  per-sample SHAP values

## Install

```bash
# Core only — numpy, pandas, networkx, plotly, scikit-learn, pyyaml
uv add concept-graph-xai

# With the SHAP adapter
uv add 'concept-graph-xai[shap]'

# With static PNG export via kaleido
uv add 'concept-graph-xai[png]'

# Everything you need to run the example notebook
uv add 'concept-graph-xai[notebook]'
```

For development, clone and `uv sync --all-extras --dev`.

## Hello world

A self-contained end-to-end check — synthetic data, sklearn model,
trivial concept tree, one sunburst — so you can confirm the install
in under 30 seconds.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from concept_graph_xai import (
    ConceptGraph, importance_sum, sunburst,
)
from concept_graph_xai.adapters import from_feature_importances_

# 1. Synthetic data
rng = np.random.default_rng(42)
X = pd.DataFrame({
    "age": rng.integers(18, 80, size=1000),
    "monthly_income": rng.normal(5000, 1500, size=1000),
    "debt_ratio": rng.uniform(0, 1, size=1000),
    "n_30_dpd": rng.poisson(0.2, size=1000),
})
y = (X["debt_ratio"] > 0.7).astype(int)

# 2. Model
model = GradientBoostingClassifier(random_state=42).fit(X, y)

# 3. Concept tree
graph = ConceptGraph.from_dict({
    "Risk": {
        "Demographics": ["age"],
        "Income":       ["monthly_income", "debt_ratio"],
        "Behaviour":    ["n_30_dpd"],
    },
})

# 4. Concept-level importance from sklearn's feature_importances_
importances, feature_names = from_feature_importances_(
    model, X.columns.tolist()
)
imp = importance_sum(graph, feature_names, importances)

# 5. One sunburst
sunburst(graph, imp, value="importance_sum",
         title="Concept importance").show()
```

If you see a three-sector sunburst (*Demographics*, *Income*,
*Behaviour*) with *Income* dominating, you are set up.

## Where to go next

- **[Tour](tour.md)** — the same scenario the example notebook
  walks, narrated end-to-end (training, SHAP, all eight question
  views, PNG export).
- **[Concept Graphs](user-guide/concept-graphs.md)** — the data
  model: dicts, YAML, NetworkX, metadata, validation.
- **User Guide** — one page per workflow question
  ([structure](user-guide/structure.md),
  [composition](user-guide/composition.md),
  [per-prediction](user-guide/per-prediction.md),
  [concept-design](user-guide/concept-design.md),
  [cohort](user-guide/cohort.md),
  [fairness](user-guide/fairness.md),
  [robustness](user-guide/robustness.md)).
- **[API Reference](api/index.md)** — full function-level docs.
- **Example notebook** —
  [`examples/01_credit_risk_walkthrough.ipynb`](https://github.com/wlazlod/concept-graph-xai/blob/main/examples/01_credit_risk_walkthrough.ipynb)
  runs end-to-end on the Give Me Some Credit Kaggle dataset.
