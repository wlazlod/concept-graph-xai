"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from concept_graph_xai import ConceptGraph
from tests.fixtures.credit_risk_toy import ToyDataset, make_dataset, make_graph

# ---------------------------------------------------------------------------
# Credit-risk toy dataset — used by tests that need a "real" tree + a fitted
# model (test_metrics, test_plotting). Session-scoped: fit happens once.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def toy() -> ToyDataset:
    return make_dataset(n=600, seed=0)


@pytest.fixture(scope="session")
def graph() -> ConceptGraph:
    return make_graph()


@pytest.fixture(scope="session")
def fitted_model(toy: ToyDataset):
    X_train, X_test, y_train, y_test = train_test_split(
        toy.X, toy.y, test_size=0.3, random_state=0, stratify=toy.y
    )
    model = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=0)
    model.fit(X_train, y_train)
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Small synthetic graph + SHAP — used by the v0.4/v0.5/v0.6 test files that
# only need a tiny tree to exercise per-concept aggregation logic. Each
# fixture has a single canonical definition here instead of being repeated
# across ~7 test files. Tests that need a different N or shape can still
# define a local fixture with the same name to override.
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_graph() -> ConceptGraph:
    """Two-branch, five-feature tree: Risk -> {Income, Behaviour}."""

    return ConceptGraph.from_dict(
        {"Risk": {"Income": ["x1", "x2"], "Behaviour": ["y1", "y2", "y3"]}}
    )


@pytest.fixture
def shap_arr() -> tuple[list[str], np.ndarray]:
    """``(feature_names, shap_values)`` aligned to ``simple_graph``.

    60 rows × 5 features, seeded so every test that doesn't override gets
    deterministic numbers.
    """

    names = ["x1", "x2", "y1", "y2", "y3"]
    arr = np.random.default_rng(0).standard_normal((60, len(names)))
    return names, arr


@pytest.fixture
def segments_series() -> pd.Series:
    """Three balanced random groups aligned to ``shap_arr`` (60 rows)."""

    return pd.Series(np.random.default_rng(0).choice(["A", "B", "C"], size=60))
