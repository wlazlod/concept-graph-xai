"""Tests for v0.3 missingness (P13/P15b) and coherence (P16) metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from concept_graph_xai import (
    ConceptGraph,
    coherence_importance,
    column_missing_rate,
    joint_missing_rate,
)


@pytest.fixture
def graph_with_missing() -> tuple[ConceptGraph, pd.DataFrame]:
    graph = ConceptGraph.from_dict(
        {"Root": {"A": ["a1", "a2"], "B": ["b1", "b2"]}}
    )
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["a1", "a2", "b1", "b2"])
    # Make 30 rows have BOTH A features missing
    block_rows = rng.choice(n, size=30, replace=False)
    X.loc[block_rows, ["a1", "a2"]] = np.nan
    # 10 rows: only b1 missing
    X.loc[rng.choice(n, size=10, replace=False), "b1"] = np.nan
    return graph, X


def test_column_missing_rate_is_per_feature(graph_with_missing) -> None:
    graph, X = graph_with_missing
    df = column_missing_rate(graph, X)
    a1_rate = df.loc[df["name"] == "a1", "column_missing_rate"].iloc[0]
    assert 0.25 <= a1_rate <= 0.35


def test_joint_missing_rate_only_when_all_features_missing(graph_with_missing) -> None:
    graph, X = graph_with_missing
    df = joint_missing_rate(graph, X)
    a_rate = df.loc[df["name"] == "A", "joint_missing_rate"].iloc[0]
    b_rate = df.loc[df["name"] == "B", "joint_missing_rate"].iloc[0]
    root_rate = df.loc[df["name"] == "Root", "joint_missing_rate"].iloc[0]
    assert 0.25 <= a_rate <= 0.35
    assert b_rate < 0.05
    assert root_rate == 0.0


def test_joint_missing_rate_for_features_equals_column_rate(graph_with_missing) -> None:
    graph, X = graph_with_missing
    jmr = joint_missing_rate(graph, X)
    cmr = column_missing_rate(graph, X)
    for feat in ["a1", "a2", "b1", "b2"]:
        joint = jmr.loc[jmr["name"] == feat, "joint_missing_rate"].iloc[0]
        col = cmr.loc[cmr["name"] == feat, "column_missing_rate"].iloc[0]
        assert joint == pytest.approx(col)


def test_coherence_importance_assigns_quadrants() -> None:
    graph = ConceptGraph.from_dict(
        {"Root": {"Coherent": ["c1", "c2"], "Incoherent": ["i1", "i2"]}}
    )
    rng = np.random.default_rng(0)
    n = 200
    base = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "c1": base + 0.05 * rng.standard_normal(n),
            "c2": base + 0.05 * rng.standard_normal(n),
            "i1": rng.standard_normal(n),
            "i2": rng.standard_normal(n),
        }
    )
    importances = np.array([1.0, 1.0, 0.05, 0.05])
    df = coherence_importance(graph, X, ["c1", "c2", "i1", "i2"], importances)
    coh_quad = df.loc[df["name"] == "Coherent", "quadrant"].iloc[0]
    inc_quad = df.loc[df["name"] == "Incoherent", "quadrant"].iloc[0]
    assert coh_quad == "well_designed"
    assert inc_quad == "noise"


def test_coherence_importance_carries_thresholds_in_attrs() -> None:
    graph = ConceptGraph.from_dict({"Root": ["a", "b"]})
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    df = coherence_importance(graph, X, ["a", "b"], np.array([0.5, 0.5]))
    assert "coherence_threshold" in df.attrs
    assert "importance_threshold" in df.attrs
    assert "method" in df.attrs
