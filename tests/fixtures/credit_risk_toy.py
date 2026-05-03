"""Tiny synthetic credit-risk-shaped dataset for tests and the README example.

The feature names are intentionally chosen so they slot naturally into a
business-concept tree (Demographics, Behaviour > Delinquency / Utilization,
Income). Used for:

* unit tests that need a real model + SHAP/permutation values
* README quickstart
* a no-network fallback inside the example notebook
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from concept_graph_xai.graph import ConceptGraph

FEATURE_NAMES: tuple[str, ...] = (
    "age",
    "n_dependents",
    "monthly_income",
    "n_30_59_dpd",
    "n_60_89_dpd",
    "n_90_plus_dpd",
    "revolving_utilization",
    "debt_ratio",
    "noise_a",
    "noise_b",
)


CONCEPT_TREE: dict[str, object] = {
    "Risk": {
        "Demographics": {
            "Age": ["age"],
            "Family": ["n_dependents"],
        },
        "Income": ["monthly_income", "debt_ratio"],
        "Behaviour": {
            "Delinquency": ["n_30_59_dpd", "n_60_89_dpd", "n_90_plus_dpd"],
            "Utilization": ["revolving_utilization"],
        },
        "Unused": ["noise_a", "noise_b"],
    }
}


@dataclass
class ToyDataset:
    X: pd.DataFrame
    y: np.ndarray
    feature_names: list[str]


def make_dataset(n: int = 2000, seed: int = 42) -> ToyDataset:
    """Generate a small synthetic dataset with a known business-meaningful signal."""

    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, n).clip(18, 90)
    n_dependents = rng.poisson(0.8, n)
    monthly_income = np.exp(rng.normal(8.6, 0.45, n))
    revolving_utilization = rng.beta(2, 5, n)
    debt_ratio = rng.beta(2, 8, n) * 2
    n_30_59_dpd = rng.poisson(0.15, n)
    n_60_89_dpd = rng.poisson(0.08, n)
    n_90_plus_dpd = rng.poisson(0.05, n)
    noise_a = rng.standard_normal(n)
    noise_b = rng.standard_normal(n)

    logit = (
        -2.5
        + 1.4 * (n_30_59_dpd + 1.5 * n_60_89_dpd + 2.0 * n_90_plus_dpd)
        + 1.1 * revolving_utilization
        + 0.7 * debt_ratio
        - 0.6 * np.log1p(monthly_income / 1000.0)
        - 0.02 * (age - 40)
        + 0.05 * n_dependents
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < p).astype(int)

    X = pd.DataFrame(
        {
            "age": age,
            "n_dependents": n_dependents,
            "monthly_income": monthly_income,
            "n_30_59_dpd": n_30_59_dpd,
            "n_60_89_dpd": n_60_89_dpd,
            "n_90_plus_dpd": n_90_plus_dpd,
            "revolving_utilization": revolving_utilization,
            "debt_ratio": debt_ratio,
            "noise_a": noise_a,
            "noise_b": noise_b,
        }
    )
    return ToyDataset(X=X, y=y, feature_names=list(FEATURE_NAMES))


def make_graph() -> ConceptGraph:
    return ConceptGraph.from_dict(CONCEPT_TREE)
