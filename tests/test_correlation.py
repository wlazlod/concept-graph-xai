"""Tests for the v0.3 correlation metrics (P14, P15a, P17)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from concept_graph_xai import (
    ConceptGraph,
    feature_correlation,
    nullity_correlation,
    shap_correlation,
)
from concept_graph_xai.metrics._common import block_boundaries


@pytest.fixture
def small_graph_and_X() -> tuple[ConceptGraph, pd.DataFrame]:
    graph = ConceptGraph.from_dict(
        {"Root": {"A": ["a1", "a2", "a3"], "B": ["b1", "b2"]}}
    )
    rng = np.random.default_rng(0)
    n = 200
    a_block = rng.standard_normal((n, 3))
    a_block[:, 1] = a_block[:, 0] + 0.05 * rng.standard_normal(n)
    a_block[:, 2] = a_block[:, 0] + 0.05 * rng.standard_normal(n)
    b_block = rng.standard_normal((n, 2)) * 0.1
    X = pd.DataFrame(
        np.hstack([a_block, b_block]),
        columns=["a1", "a2", "a3", "b1", "b2"],
    )
    return graph, X


def test_block_boundaries_match_graph_order(small_graph_and_X) -> None:
    graph, _ = small_graph_and_X
    blocks = block_boundaries(graph)
    paths = [b[0] for b in blocks]
    assert "Root" in paths
    assert "Root/A" in paths
    assert "Root/B" in paths
    a_block = next(b for b in blocks if b[0] == "Root/A")
    assert a_block[2] - a_block[1] == 3
    b_block = next(b for b in blocks if b[0] == "Root/B")
    assert b_block[2] - b_block[1] == 2


def test_feature_correlation_matrix_shape(small_graph_and_X) -> None:
    graph, X = small_graph_and_X
    result = feature_correlation(graph, X, method="spearman")
    assert result.matrix.shape == (5, 5)
    assert list(result.matrix.columns) == ["a1", "a2", "a3", "b1", "b2"]
    assert result.method == "spearman"


def test_feature_correlation_block_stats_show_high_within_a(small_graph_and_X) -> None:
    graph, X = small_graph_and_X
    result = feature_correlation(graph, X)
    stats = result.block_stats.set_index("concept_path").to_dict("index")
    assert stats["Root/A"]["mean_abs"] > 0.85
    assert stats["Root/B"]["mean_abs"] < 0.5


def test_nullity_correlation_when_features_share_missing_pattern() -> None:
    graph = ConceptGraph.from_dict({"Root": {"A": ["a1", "a2"], "B": ["b1"]}})
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a1", "a2", "b1"])
    rows_to_blank = rng.choice(n, size=80, replace=False)
    X.loc[rows_to_blank, ["a1", "a2"]] = np.nan
    result = nullity_correlation(graph, X)
    stats = result.block_stats.set_index("concept_path").to_dict("index")
    assert stats["Root/A"]["mean_abs"] > 0.95


def test_nullity_correlation_handles_no_missing_data() -> None:
    graph = ConceptGraph.from_dict({"Root": ["a", "b"]})
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = nullity_correlation(graph, X)
    assert (result.matrix.to_numpy() == 0.0).all()


def test_shap_correlation_picks_up_signed_redundancy() -> None:
    graph = ConceptGraph.from_dict({"Root": {"A": ["a1", "a2"], "B": ["b"]}})
    rng = np.random.default_rng(2)
    n = 250
    base = rng.standard_normal(n)
    shap_values = np.column_stack([base, base + 0.05 * rng.standard_normal(n), rng.standard_normal(n)])
    result = shap_correlation(graph, ["a1", "a2", "b"], shap_values)
    stats = result.block_stats.set_index("concept_path").to_dict("index")
    assert stats["Root/A"]["mean_abs"] > 0.9


def test_correlation_method_argument_is_respected(small_graph_and_X) -> None:
    graph, X = small_graph_and_X
    pearson = feature_correlation(graph, X, method="pearson")
    spearman = feature_correlation(graph, X, method="spearman")
    assert pearson.method == "pearson"
    assert spearman.method == "spearman"
    assert not np.allclose(pearson.matrix.to_numpy(), spearman.matrix.to_numpy())
