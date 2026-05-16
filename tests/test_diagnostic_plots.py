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
    z = np.asarray(fig.data[0].z, dtype=float)
    assert z.shape == (5, 5)
    # Correlations must live in [-1, 1] and the diagonal should be 1.
    assert ((z >= -1.0) & (z <= 1.0)).all()
    np.testing.assert_allclose(np.diag(z), 1.0, atol=1e-9)


def test_correlation_block_renders_for_nullity_correlation(graph, X) -> None:
    Xm = X.copy()
    Xm.iloc[:30, :2] = np.nan
    res = nullity_correlation(graph, Xm)
    fig = correlation_block(res, title="nullity")
    assert fig.data
    assert fig.data[0].type == "heatmap"
    z = np.asarray(fig.data[0].z, dtype=float)
    assert ((z >= -1.0) & (z <= 1.0)).all()


def test_correlation_block_renders_for_shap_correlation(graph, X) -> None:
    rng = np.random.default_rng(1)
    shap_values = rng.standard_normal((len(X), 5))
    res = shap_correlation(graph, list(X.columns), shap_values)
    fig = correlation_block(res, title="shap")
    assert fig.data
    assert fig.data[0].type == "heatmap"
    z = np.asarray(fig.data[0].z, dtype=float)
    assert ((z >= -1.0) & (z <= 1.0)).all()
    np.testing.assert_allclose(np.diag(z), 1.0, atol=1e-9)


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


def test_regulatory_tag_overlay_same_tag_same_color_and_untagged_fallback() -> None:
    """Two nodes with the same tag → same colour; missing tag → untagged_color."""

    import networkx as nx

    g = nx.DiGraph()
    g.add_node("Root", kind="concept", metadata={})
    g.add_node("A", kind="concept", metadata={"tag": "PII"})
    g.add_node("B", kind="concept", metadata={"tag": "PII"})
    g.add_node("C", kind="concept", metadata={})  # no tag → untagged
    g.add_node("a1", kind="feature", metadata={"tag": "PII"})
    g.add_node("b1", kind="feature", metadata={"tag": "non-PII"})
    g.add_node("c1", kind="feature", metadata={})  # no tag → untagged
    g.add_edge("Root", "A")
    g.add_edge("Root", "B")
    g.add_edge("Root", "C")
    g.add_edge("A", "a1")
    g.add_edge("B", "b1")
    g.add_edge("C", "c1")
    graph = ConceptGraph.from_networkx(g, root="Root")
    fig = regulatory_tag_overlay(graph, tag_key="tag", untagged_color="#abcdef")
    labels = list(fig.data[0].labels)
    colors = list(fig.data[0].marker.colors)
    label_to_color = dict(zip(labels, colors, strict=True))
    # Same PII tag → same colour for A, B, and a1.
    assert label_to_color["A"] == label_to_color["B"] == label_to_color["a1"]
    # Different tag → different colour for b1.
    assert label_to_color["b1"] != label_to_color["A"]
    # Untagged nodes fall back to untagged_color.
    assert label_to_color["C"] == "#abcdef"
    assert label_to_color["c1"] == "#abcdef"
