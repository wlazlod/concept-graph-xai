"""Unit tests for the metric layer."""

from __future__ import annotations

import numpy as np
import pytest

from concept_graph_xai import (
    ConceptGraph,
    auc_drop,
    feature_counts,
    importance_sum,
    utilization,
)


@pytest.fixture
def small_graph() -> ConceptGraph:
    return ConceptGraph.from_dict(
        {"Root": {"A": ["f1", "f2"], "B": ["f3", "f4"]}}
    )


def test_feature_counts_aggregates_descendants(small_graph: ConceptGraph) -> None:
    df = feature_counts(small_graph)
    by_name = dict(zip(df["name"], df["count"], strict=True))
    assert by_name["Root"] == 4
    assert by_name["A"] == 2
    assert by_name["B"] == 2
    assert by_name["f1"] == 1


def test_importance_sum_with_1d_array(small_graph: ConceptGraph) -> None:
    importances = np.array([1.0, 2.0, 3.0, 4.0])
    df = importance_sum(small_graph, ["f1", "f2", "f3", "f4"], importances)
    by_name = dict(zip(df["name"], df["importance_sum"], strict=True))
    assert by_name["A"] == 3.0
    assert by_name["B"] == 7.0
    assert by_name["Root"] == 10.0


def test_importance_sum_aggregates_2d_with_abs_mean(small_graph: ConceptGraph) -> None:
    rng = np.random.default_rng(0)
    n_samples, n_features = 50, 4
    values = rng.standard_normal((n_samples, n_features))
    df = importance_sum(small_graph, ["f1", "f2", "f3", "f4"], values, signed=False)
    by_name = dict(zip(df["name"], df["importance_sum"], strict=True))
    expected_root = float(np.abs(values).mean(axis=0).sum())
    assert by_name["Root"] == pytest.approx(expected_root)


def test_utilization_marks_used_above_threshold(small_graph: ConceptGraph) -> None:
    importances = np.array([0.5, 0.0, 0.0, 0.4])
    df = utilization(small_graph, ["f1", "f2", "f3", "f4"], importances, threshold=0.1)
    by_name = dict(zip(df["name"], df["is_used"], strict=True))
    assert by_name["f1"] is np.True_ or by_name["f1"] is True
    assert by_name["f2"] is np.False_ or by_name["f2"] is False
    assert by_name["A"]
    assert by_name["B"]
    assert by_name["Root"]
    assert by_name["f3"] is np.False_ or by_name["f3"] is False


def test_utilization_all_zero_yields_unused(small_graph: ConceptGraph) -> None:
    importances = np.zeros(4)
    df = utilization(small_graph, ["f1", "f2", "f3", "f4"], importances, threshold=0.0)
    assert not df["is_used"].any()


def test_auc_drop_root_feature_count_matches_total(graph, fitted_model, toy) -> None:
    """Regression: the root must carry the full feature_count even when skip_root=True.

    Without it, plotting.auc_drop_map's branchvalues='total' silently renders
    an empty sunburst because the parent is smaller than the sum of children.
    """

    df = auc_drop(
        graph,
        fitted_model["model"],
        fitted_model["X_test"],
        fitted_model["y_test"],
        feature_names=toy.feature_names,
        strategy="permutation",
        n_repeats=2,
        random_state=0,
    )
    root_count = int(df.loc[df["name"] == graph.root, "feature_count"].iloc[0])
    direct_children = [c for c in graph.children_of(graph.root)]
    children_sum = int(
        df.loc[df["name"].isin(direct_children), "feature_count"].sum()
    )
    assert root_count == children_sum
    assert root_count > 0


def test_auc_drop_permutation_returns_expected_columns(graph, fitted_model, toy) -> None:
    df = auc_drop(
        graph,
        fitted_model["model"],
        fitted_model["X_test"],
        fitted_model["y_test"],
        feature_names=toy.feature_names,
        strategy="permutation",
        n_repeats=2,
        random_state=0,
    )
    for col in [
        "auc_drop_mean",
        "auc_drop_std",
        "ablated_score_mean",
        "baseline_score",
        "feature_count",
        "strategy",
    ]:
        assert col in df.columns
    delinquency_drop = df.loc[df["name"] == "Delinquency", "auc_drop_mean"].iloc[0]
    unused_drop = df.loc[df["name"] == "Unused", "auc_drop_mean"].iloc[0]
    assert delinquency_drop > unused_drop


def test_auc_drop_shap_marginal_with_synthetic_shap(graph, fitted_model, toy) -> None:
    model = fitted_model["model"]
    X_test = fitted_model["X_test"]
    y_test = fitted_model["y_test"]
    base_pred = model.predict_proba(X_test)[:, 1]
    rng = np.random.default_rng(1)
    shap_values = rng.standard_normal((len(X_test), len(toy.feature_names))) * 0.05

    df = auc_drop(
        graph,
        model,
        X_test,
        y_test,
        feature_names=toy.feature_names,
        strategy="shap_marginal",
        shap_values=shap_values,
        base_predictions=base_pred,
    )
    assert "auc_drop_mean" in df.columns
    assert df["strategy"].iloc[0] == "shap_marginal"


def test_auc_drop_retrain_with_callable(graph, fitted_model, toy) -> None:
    from sklearn.ensemble import GradientBoostingClassifier

    def train_fn(X, y):
        m = GradientBoostingClassifier(n_estimators=20, max_depth=2, random_state=0)
        m.fit(X, y)
        return m

    df = auc_drop(
        graph,
        fitted_model["model"],
        fitted_model["X_test"],
        fitted_model["y_test"],
        feature_names=toy.feature_names,
        strategy="retrain",
        train_fn=train_fn,
        X_train=fitted_model["X_train"],
        y_train=fitted_model["y_train"],
    )
    assert "auc_drop_mean" in df.columns
    assert df["strategy"].iloc[0] == "retrain"
    assert (df["auc_drop_mean"].dropna() != 0).any()
