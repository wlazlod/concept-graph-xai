"""Tests for v0.4 local-explanation features (P1 beeswarm, P5 waterfall)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from concept_graph_xai import (
    ConceptGraph,
    ConceptPredictionExplainer,
    concept_violin,
)


@pytest.fixture
def shap_arr(simple_graph: ConceptGraph) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    n = 50
    names = ["x1", "x2", "y1", "y2", "y3"]
    arr = rng.standard_normal((n, len(names)))
    return names, arr


def test_concept_violin_emits_one_trace_per_concept(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    fig = concept_violin(simple_graph, names, arr)
    assert fig.data
    n_concepts = len([c for c in simple_graph.concepts() if c != simple_graph.root])
    assert len(fig.data) == n_concepts


def test_concept_violin_respects_max_concepts(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    fig = concept_violin(simple_graph, names, arr, max_concepts=2)
    assert len(fig.data) == 2


def test_concept_violin_with_features(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    fig = concept_violin(simple_graph, names, arr, only_concepts=False)
    assert len(fig.data) >= len(names)


def test_concept_violin_rejects_shape_mismatch(simple_graph) -> None:
    arr = np.zeros((10, 3))
    with pytest.raises(ValueError, match="cols"):
        concept_violin(simple_graph, ["x1", "x2", "y1", "y2", "y3"], arr)


def test_prediction_explainer_breakdown_has_one_row_per_concept(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    X = pd.DataFrame(np.zeros_like(arr), columns=names)
    exp = ConceptPredictionExplainer(simple_graph, model=None, X=X, shap_values=arr, base_value=0.0)
    df = exp.breakdown(0, depth=1)
    assert set(df["name"]) == {"Income", "Behaviour"}
    assert df["shap_sum"].notna().all()


def test_prediction_explainer_breakdown_depth_with_subconcepts() -> None:
    simple_graph = ConceptGraph.from_dict(
        {"Risk": {"Demographics": {"Age": ["age"], "Family": ["dep"]}}}
    )
    names = ["age", "dep"]
    arr = np.array([[0.5, -0.2], [0.1, 0.3]])
    X = pd.DataFrame(arr, columns=names)
    exp = ConceptPredictionExplainer(simple_graph, model=None, X=X, shap_values=arr, base_value=0.0)
    df = exp.breakdown(0, depth=2)
    assert set(df["name"]) == {"Age", "Family"}


def test_prediction_explainer_raises_for_missing_depth(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    X = pd.DataFrame(np.zeros_like(arr), columns=names)
    exp = ConceptPredictionExplainer(simple_graph, model=None, X=X, shap_values=arr, base_value=0.0)
    with pytest.raises(ValueError, match="no concepts at depth"):
        exp.breakdown(0, depth=3)


def test_prediction_explainer_waterfall_renders(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    X = pd.DataFrame(np.zeros_like(arr), columns=names)
    exp = ConceptPredictionExplainer(
        simple_graph, model=None, X=X, shap_values=arr, base_value=-1.0
    )
    fig = exp.waterfall(0, depth=1)
    assert fig.data
    assert fig.data[0].type == "waterfall"
    measures = list(fig.data[0].measure)
    assert measures[0] == "absolute"
    assert measures[-1] == "total"


def test_prediction_explainer_resolves_label(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    idx = pd.Index([f"row_{i}" for i in range(arr.shape[0])])
    X = pd.DataFrame(np.zeros_like(arr), columns=names, index=idx)
    exp = ConceptPredictionExplainer(simple_graph, model=None, X=X, shap_values=arr, base_value=0.0)
    fig = exp.waterfall("row_3", depth=1)
    assert fig.data
    assert "row_3" in fig.layout.title.text


def test_prediction_explainer_rejects_bad_shape() -> None:
    g = ConceptGraph.from_dict({"R": ["a", "b"]})
    X = pd.DataFrame({"a": [0.0], "b": [0.0]})
    with pytest.raises(ValueError, match="2D"):
        ConceptPredictionExplainer(g, model=None, X=X, shap_values=np.zeros(2), base_value=0.0)
