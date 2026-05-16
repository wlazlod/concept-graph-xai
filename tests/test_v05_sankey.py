"""Tests for v0.5 concept_sankey (P4)."""

from __future__ import annotations

import numpy as np
import pytest

from concept_graph_xai import ConceptGraph, concept_sankey


@pytest.fixture
def graph() -> ConceptGraph:
    return ConceptGraph.from_dict(
        {"Risk": {"Income": ["x1", "x2"], "Behaviour": ["y1", "y2", "y3"]}}
    )


@pytest.fixture
def shap_arr() -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    names = ["x1", "x2", "y1", "y2", "y3"]
    arr = rng.standard_normal((50, len(names)))
    return names, arr


def test_concept_sankey_renders(graph, shap_arr) -> None:
    names, arr = shap_arr
    fig = concept_sankey(graph, names, arr)
    assert fig.data
    assert fig.data[0].type == "sankey"


def test_concept_sankey_node_layout_two_level_graph(graph, shap_arr) -> None:
    # Two-level graph: features connect directly to top-level concepts.
    names, arr = shap_arr
    fig = concept_sankey(graph, names, arr)
    labels = list(fig.data[0].node.label)
    # 5 features + 2 top-level concepts (Income, Behaviour) + 2 outcome = 9
    assert len(labels) == 9
    assert {"Income", "Behaviour", "+ outcome", "- outcome"}.issubset(set(labels))


def test_concept_sankey_includes_intermediate_concepts() -> None:
    # Three-level graph: features → sub-concept → top-level concept.
    graph = ConceptGraph.from_dict(
        {
            "Risk": {
                "Demographics": {"Age": ["age"], "Family": ["dep"]},
                "Behaviour": {"Delinquency": ["d1", "d2"]},
            }
        }
    )
    names = ["age", "dep", "d1", "d2"]
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((30, 4))
    fig = concept_sankey(graph, names, arr)
    labels = list(fig.data[0].node.label)
    # All hierarchy levels must be present: features + sub-concepts + top-level
    for required in (
        "age",
        "dep",
        "d1",
        "d2",
        "Age",
        "Family",
        "Delinquency",
        "Demographics",
        "Behaviour",
        "+ outcome",
        "- outcome",
    ):
        assert required in labels, f"missing {required!r}"


def test_concept_sankey_link_count_matches_hierarchy(graph, shap_arr) -> None:
    names, arr = shap_arr
    fig = concept_sankey(graph, names, arr)
    sources = list(fig.data[0].link.source)
    # 5 feature→concept + up to 4 concept→outcome = 5..9
    assert 5 <= len(sources) <= 9
    assert len(sources) == len(fig.data[0].link.target)


def test_concept_sankey_max_features_per_concept_caps(graph, shap_arr) -> None:
    names, arr = shap_arr
    fig = concept_sankey(graph, names, arr, max_features_per_concept=1)
    labels = list(fig.data[0].node.label)
    # 1 feature per concept * 2 concepts = 2 features + 2 concepts + 2 outcomes = 6
    assert len(labels) == 6


def test_concept_sankey_rejects_bad_input(graph, shap_arr) -> None:
    names, arr = shap_arr
    with pytest.raises(ValueError, match="2D"):
        concept_sankey(graph, names, arr.sum(axis=1))
    with pytest.raises(ValueError, match="cols"):
        concept_sankey(graph, names[:2], arr)


def test_concept_sankey_link_values_are_nonnegative(graph, shap_arr) -> None:
    names, arr = shap_arr
    fig = concept_sankey(graph, names, arr)
    values = list(fig.data[0].link.value)
    assert all(v >= 0 for v in values)


def test_concept_sankey_outcome_split_matches_signed_flow(graph, shap_arr) -> None:
    names, arr = shap_arr
    fig = concept_sankey(graph, names, arr)
    labels = list(fig.data[0].node.label)
    sources = list(fig.data[0].link.source)
    targets = list(fig.data[0].link.target)
    values = list(fig.data[0].link.value)
    pos_idx = labels.index("+ outcome")
    neg_idx = labels.index("- outcome")
    pos_total = sum(v for s, t, v in zip(sources, targets, values, strict=True) if t == pos_idx)
    neg_total = sum(v for s, t, v in zip(sources, targets, values, strict=True) if t == neg_idx)
    income_cols = [names.index(f) for f in ("x1", "x2")]
    behav_cols = [names.index(f) for f in ("y1", "y2", "y3")]
    expected_pos = float(
        np.maximum(arr[:, income_cols].sum(axis=1), 0).sum()
        + np.maximum(arr[:, behav_cols].sum(axis=1), 0).sum()
    )
    expected_neg = float(
        np.maximum(-arr[:, income_cols].sum(axis=1), 0).sum()
        + np.maximum(-arr[:, behav_cols].sum(axis=1), 0).sum()
    )
    assert pos_total == pytest.approx(expected_pos)
    assert neg_total == pytest.approx(expected_neg)


def test_concept_sankey_subconcept_outflow_lte_inflow_under_cancellation() -> None:
    # Sub-concept Delinquency wraps two features that perfectly anti-correlate.
    # Inflow = sum |SHAP[d1]| + sum |SHAP[d2]|; outflow = sum |SHAP[d1] + SHAP[d2]| ≈ 0.
    graph = ConceptGraph.from_dict({"Risk": {"Behaviour": {"Delinquency": ["d1", "d2"]}}})
    names = ["d1", "d2"]
    arr = np.array([[1.0, -1.0], [2.0, -2.0], [-1.5, 1.5]])
    fig = concept_sankey(graph, names, arr)
    labels = list(fig.data[0].node.label)
    sources = list(fig.data[0].link.source)
    targets = list(fig.data[0].link.target)
    values = list(fig.data[0].link.value)
    delinq_idx = labels.index("Delinquency")
    behav_idx = labels.index("Behaviour")
    inflow = sum(v for s, t, v in zip(sources, targets, values, strict=True) if t == delinq_idx)
    outflow = sum(
        v
        for s, t, v in zip(sources, targets, values, strict=True)
        if s == delinq_idx and t == behav_idx
    )
    assert outflow < inflow, "cancelling features must shrink the sub-concept outflow"
    assert outflow == pytest.approx(0.0, abs=1e-9)


def test_concept_sankey_features_in_dfs_order_groups_siblings() -> None:
    # Features should appear in DFS preorder so siblings under the same
    # parent are adjacent in the node array (and therefore vertically in the
    # Sankey).
    graph = ConceptGraph.from_dict(
        {
            "Risk": {
                "Demographics": {"Age": ["age"], "Family": ["dep"]},
                "Behaviour": ["b1", "b2"],
            }
        }
    )
    names = ["age", "dep", "b1", "b2"]
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((20, 4))
    fig = concept_sankey(graph, names, arr)
    labels = list(fig.data[0].node.label)
    # Demographics-side features must precede Behaviour-side features
    assert labels.index("age") < labels.index("b1")
    assert labels.index("dep") < labels.index("b1")
    # Within Demographics, Age subtree before Family (DFS preorder)
    assert labels.index("age") < labels.index("dep")
    # Behaviour features adjacent (no other features between them)
    assert labels.index("b2") - labels.index("b1") == 1


def test_concept_sankey_pins_explicit_node_positions() -> None:
    # Explicit x/y must be set per node so Plotly's link-crossing-minimization
    # cannot reorder the layout away from the ontological grouping.
    graph = ConceptGraph.from_dict(
        {
            "Risk": {
                "Demographics": {"Age": ["age"], "Family": ["dep"]},
                "Behaviour": ["b1", "b2"],
            }
        }
    )
    names = ["age", "dep", "b1", "b2"]
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((30, 4))
    fig = concept_sankey(graph, names, arr)
    node = fig.data[0].node
    xs = list(node.x)
    ys = list(node.y)
    labels = list(node.label)
    assert len(xs) == len(labels)
    assert len(ys) == len(labels)
    # Features sit at the leftmost x; outcomes at the rightmost
    feature_xs = {xs[labels.index(f)] for f in names}
    assert len(feature_xs) == 1
    feature_x = feature_xs.pop()
    plus_x = xs[labels.index("+ outcome")]
    minus_x = xs[labels.index("- outcome")]
    assert feature_x < plus_x == minus_x
    # Top-level concepts (Demographics, Behaviour) sit just left of the outcomes
    top_x = xs[labels.index("Demographics")]
    assert top_x < plus_x
    # Sub-concepts (Age, Family) sit between features and top-level concepts
    sub_x = xs[labels.index("Age")]
    assert feature_x < sub_x < top_x
    assert xs[labels.index("Family")] == sub_x
    # Within the feature tier, DFS preorder enforces y order.
    # Plotly Sankey y=0 is the TOP, y=1 is the BOTTOM, so first-in-DFS
    # gets the smallest y.
    assert ys[labels.index("age")] < ys[labels.index("dep")]
    assert ys[labels.index("dep")] < ys[labels.index("b1")]
    assert ys[labels.index("b1")] < ys[labels.index("b2")]


def test_concept_sankey_concept_tiers_deepest_first() -> None:
    # Sub-concepts (depth 2) must come BEFORE top-level concepts (depth 1)
    # in the node array so Plotly snaps them to a tier left of the top-level.
    graph = ConceptGraph.from_dict({"Risk": {"Demographics": {"Age": ["age"], "Family": ["dep"]}}})
    names = ["age", "dep"]
    arr = np.random.default_rng(0).standard_normal((10, 2))
    fig = concept_sankey(graph, names, arr)
    labels = list(fig.data[0].node.label)
    assert labels.index("Age") < labels.index("Demographics")
    assert labels.index("Family") < labels.index("Demographics")


def test_concept_sankey_skips_features_directly_under_root() -> None:
    # Pathological: feature directly under root with no concept layer.
    graph = ConceptGraph.from_dict({"Risk": ["a", "b"]})
    arr = np.random.default_rng(0).standard_normal((10, 2))
    with pytest.raises(ValueError, match="non-zero SHAP magnitude under a concept"):
        concept_sankey(graph, ["a", "b"], arr)
