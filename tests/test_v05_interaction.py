"""Tests for v0.5 concept_interaction_matrix + concept_interaction_heatmap (P3)."""

from __future__ import annotations

import numpy as np
import pytest

from concept_graph_xai import (
    ConceptGraph,
    concept_interaction_heatmap,
    concept_interaction_matrix,
)


@pytest.fixture
def graph() -> ConceptGraph:
    return ConceptGraph.from_dict(
        {"Risk": {"Income": ["x1", "x2"], "Behaviour": ["y1", "y2", "y3"]}}
    )


@pytest.fixture
def interaction_arr() -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    n, f = 50, 5
    raw = rng.standard_normal((n, f, f))
    # Symmetrize on the (i, j) axes per-sample (real shap interaction values are symmetric)
    arr = 0.5 * (raw + raw.transpose(0, 2, 1))
    return ["x1", "x2", "y1", "y2", "y3"], arr


def test_interaction_matrix_is_square_and_symmetric(graph, interaction_arr) -> None:
    names, arr = interaction_arr
    df = concept_interaction_matrix(graph, names, arr)
    assert df.shape[0] == df.shape[1]
    assert list(df.index) == list(df.columns)
    np.testing.assert_allclose(df.to_numpy(), df.to_numpy().T, atol=1e-12)


def test_interaction_matrix_drops_root_by_default(graph, interaction_arr) -> None:
    names, arr = interaction_arr
    df = concept_interaction_matrix(graph, names, arr)
    assert graph.root not in df.index
    assert {"Income", "Behaviour"}.issubset(set(df.index))


def test_interaction_matrix_only_concepts_excludes_features(graph, interaction_arr) -> None:
    names, arr = interaction_arr
    df = concept_interaction_matrix(graph, names, arr, only_concepts=True)
    assert "x1" not in df.index
    df_with_features = concept_interaction_matrix(graph, names, arr, only_concepts=False)
    assert "x1" in df_with_features.index


def test_interaction_matrix_diagonal_equals_self_block_sum(graph, interaction_arr) -> None:
    names, arr = interaction_arr
    df = concept_interaction_matrix(graph, names, arr, agg="mean_abs")
    # Income's diagonal cell = mean_n |sum_{i,j ∈ {x1,x2}} arr[n, i, j]|
    income_idxs = [names.index(f) for f in ("x1", "x2")]
    sub = arr[:, income_idxs, :][:, :, income_idxs]
    expected = float(np.abs(sub.sum(axis=(1, 2))).mean())
    assert df.loc["Income", "Income"] == pytest.approx(expected)


def test_interaction_matrix_signed_can_be_negative(graph, interaction_arr) -> None:
    names, arr = interaction_arr
    df = concept_interaction_matrix(graph, names, arr, agg="mean_signed")
    assert df.attrs["agg"] == "mean_signed"
    # The signed agg should give a different number from mean_abs for at least one cell
    df_abs = concept_interaction_matrix(graph, names, arr, agg="mean_abs")
    assert not np.allclose(df.to_numpy(), df_abs.to_numpy())


def test_interaction_matrix_rejects_bad_input(graph, interaction_arr) -> None:
    names, arr = interaction_arr
    with pytest.raises(ValueError, match="3D"):
        concept_interaction_matrix(graph, names, arr.sum(axis=0))
    with pytest.raises(ValueError, match="square"):
        concept_interaction_matrix(graph, names, arr[:, :, :3])
    with pytest.raises(ValueError, match="features"):
        concept_interaction_matrix(graph, ["x1", "x2"], arr)


def test_interaction_heatmap_renders(graph, interaction_arr) -> None:
    names, arr = interaction_arr
    df = concept_interaction_matrix(graph, names, arr)
    fig = concept_interaction_heatmap(df)
    assert fig.data
    assert fig.data[0].type == "heatmap"
    assert list(fig.data[0].x) == list(df.columns)


def test_interaction_heatmap_signed_uses_diverging_palette(graph, interaction_arr) -> None:
    names, arr = interaction_arr
    df = concept_interaction_matrix(graph, names, arr, agg="mean_signed")
    fig = concept_interaction_heatmap(df)
    assert fig.data[0].zmid == 0.0


def test_interaction_heatmap_annotations_capped() -> None:
    # Larger graph so we have several off-diagonal cells
    graph = ConceptGraph.from_dict(
        {"Risk": {"A": ["a1"], "B": ["b1"], "C": ["c1"], "D": ["d1"]}}
    )
    names = ["a1", "b1", "c1", "d1"]
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((30, 4, 4))
    arr = 0.5 * (raw + raw.transpose(0, 2, 1))
    df = concept_interaction_matrix(graph, names, arr)
    # 4 concepts → 6 off-diagonal cells in upper triangle
    fig = concept_interaction_heatmap(df, annotate_top_k=3)
    assert len(list(fig.layout.annotations)) == 3
    fig_none = concept_interaction_heatmap(df, annotate_top_k=None)
    assert len(list(fig_none.layout.annotations)) == 0
