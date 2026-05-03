"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from tests.fixtures.credit_risk_toy import ToyDataset, make_dataset, make_graph


@pytest.fixture(scope="session")
def toy() -> ToyDataset:
    return make_dataset(n=600, seed=0)


@pytest.fixture(scope="session")
def graph():
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
