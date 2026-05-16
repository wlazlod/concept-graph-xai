"""Unit tests for ConceptGraph."""

from __future__ import annotations

import pytest

from concept_graph_xai import ConceptGraph


def test_from_dict_simple_tree() -> None:
    graph = ConceptGraph.from_dict({"Root": {"A": ["f1", "f2"], "B": ["f3"]}})
    assert graph.root == "Root"
    assert set(graph.features()) == {"f1", "f2", "f3"}
    assert set(graph.concepts()) == {"Root", "A", "B"}
    assert graph.parent_of("f1") == "A"
    assert graph.descendant_features("Root") == ["f1", "f2", "f3"]


def test_from_dict_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        ConceptGraph.from_dict({"Root": {"A": ["f1"], "B": ["f1"]}})


def test_validation_concept_must_have_children() -> None:
    import networkx as nx

    g = nx.DiGraph()
    g.add_node("Root", kind="concept", metadata={})
    g.add_node("Empty", kind="concept", metadata={})
    g.add_edge("Root", "Empty")
    with pytest.raises(ValueError, match="orphan concept"):
        ConceptGraph.from_networkx(g, root="Root")


def test_validation_feature_must_be_leaf() -> None:
    import networkx as nx

    g = nx.DiGraph()
    g.add_node("Root", kind="concept", metadata={})
    g.add_node("F", kind="feature", metadata={})
    g.add_node("X", kind="feature", metadata={})
    g.add_edge("Root", "F")
    g.add_edge("F", "X")
    with pytest.raises(ValueError, match="must be a leaf"):
        ConceptGraph.from_networkx(g, root="Root")


def test_path_uses_full_root_chain() -> None:
    graph = ConceptGraph.from_dict({"Root": {"A": {"B": ["leaf"]}}})
    assert graph.path("leaf") == ("Root", "A", "B", "leaf")


def test_traversal_order_is_deterministic() -> None:
    graph = ConceptGraph.from_dict({"Root": {"A": ["f1", "f2"], "B": ["f3"]}})
    order_first = graph.nodes_in_order()
    order_second = graph.nodes_in_order()
    assert order_first == order_second
    assert order_first[0] == "Root"


def test_graph_property_is_a_snapshot() -> None:
    graph = ConceptGraph.from_dict({"Root": {"A": ["f1", "f2"]}})
    snapshot = graph.graph
    # Mutate the snapshot — the original must be untouched.
    snapshot.add_node("intruder", kind="feature", metadata={})
    snapshot.add_edge("A", "intruder")
    assert "intruder" not in graph
    assert graph.features() == ["f1", "f2"]
    # And the path cache must still be intact after re-querying.
    assert graph.path("f1") == ("Root", "A", "f1")


def test_yaml_roundtrip(tmp_path) -> None:
    from concept_graph_xai.io import dump_yaml, load_yaml

    data = {"Root": {"A": ["f1", "f2"], "B": ["f3"]}}
    path = tmp_path / "tree.yaml"
    dump_yaml(data, path)
    reloaded = load_yaml(path)
    graph = ConceptGraph.from_dict(reloaded)
    assert set(graph.features()) == {"f1", "f2", "f3"}
