"""Smoke tests for v0.3 plotting (P12, P13, P14, P15b, P16, P17)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from concept_graph_xai import (
    ConceptGraph,
    coherence_importance,
    coherence_importance_scatter,
    correlation_block,
    feature_correlation,
    joint_missing_map,
    joint_missing_rate,
    nullity_correlation,
    regulatory_tag_overlay,
    shap_correlation,
)


@pytest.fixture
def graph() -> ConceptGraph:
    return ConceptGraph.from_dict(
        {"Root": {"Income": ["x1", "x2"], "Behaviour": ["y1", "y2", "y3"]}}
    )


@pytest.fixture
def X(graph: ConceptGraph) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 150
    base = rng.standard_normal(n)
    return pd.DataFrame(
        {
            "x1": base + 0.05 * rng.standard_normal(n),
            "x2": base + 0.05 * rng.standard_normal(n),
            "y1": rng.standard_normal(n),
            "y2": rng.standard_normal(n),
            "y3": rng.standard_normal(n),
        }
    )


def test_correlation_block_renders_for_feature_correlation(graph, X) -> None:
    res = feature_correlation(graph, X)
    fig = correlation_block(res, title="features")
    assert fig.data
    assert fig.data[0].type == "heatmap"
    assert fig.data[0].z.shape == (5, 5)


def test_correlation_block_renders_for_nullity_correlation(graph, X) -> None:
    Xm = X.copy()
    Xm.iloc[:30, :2] = np.nan
    res = nullity_correlation(graph, Xm)
    fig = correlation_block(res, title="nullity")
    assert fig.data
    assert fig.data[0].type == "heatmap"


def test_correlation_block_renders_for_shap_correlation(graph, X) -> None:
    rng = np.random.default_rng(1)
    shap_values = rng.standard_normal((len(X), 5))
    res = shap_correlation(graph, list(X.columns), shap_values)
    fig = correlation_block(res, title="shap")
    assert fig.data
    assert fig.data[0].type == "heatmap"


def test_joint_missing_map_uses_rate_for_color(graph, X) -> None:
    Xm = X.copy()
    Xm.iloc[:30, :2] = np.nan
    df = joint_missing_rate(graph, Xm)
    fig = joint_missing_map(graph, df)
    assert fig.data
    assert fig.data[0].type == "sunburst"
    colors = list(fig.data[0].marker.colors)
    assert max(colors) > 0


def test_coherence_importance_scatter_renders(graph, X) -> None:
    importances = np.array([1.0, 1.0, 0.1, 0.1, 0.1])
    df = coherence_importance(graph, X, list(X.columns), importances)
    fig = coherence_importance_scatter(df)
    assert fig.data
    assert all(t.type == "scatter" for t in fig.data)


def test_regulatory_tag_overlay_uses_metadata() -> None:
    import networkx as nx

    g = nx.DiGraph()
    g.add_node("Root", kind="concept", metadata={})
    g.add_node("Income", kind="concept", metadata={"tag": "PII"})
    g.add_node("x1", kind="feature", metadata={"tag": "PII"})
    g.add_node("x2", kind="feature", metadata={"tag": "non-PII"})
    g.add_edge("Root", "Income")
    g.add_edge("Income", "x1")
    g.add_edge("Income", "x2")
    graph = ConceptGraph.from_networkx(g, root="Root")
    fig = regulatory_tag_overlay(graph, tag_key="tag")
    assert fig.data
    assert fig.data[0].type == "sunburst"
    colors = list(fig.data[0].marker.colors)
    assert len(set(colors)) >= 2
