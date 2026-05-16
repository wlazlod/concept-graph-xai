"""Tests for v0.5 bootstrap_importance + signed_concept_bar (P2)."""

from __future__ import annotations

import numpy as np
import pytest

from concept_graph_xai import ConceptGraph, bootstrap_importance, signed_concept_bar


@pytest.fixture
def graph() -> ConceptGraph:
    return ConceptGraph.from_dict(
        {"Risk": {"Income": ["x1", "x2"], "Behaviour": ["y1", "y2", "y3"]}}
    )


@pytest.fixture
def shap_arr() -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    names = ["x1", "x2", "y1", "y2", "y3"]
    arr = rng.standard_normal((100, len(names)))
    return names, arr


def test_bootstrap_importance_columns_and_attrs(graph, shap_arr) -> None:
    names, arr = shap_arr
    df = bootstrap_importance(graph, names, arr, n_bootstrap=50, random_state=0)
    for col in (
        "name",
        "kind",
        "depth",
        "parent",
        "mean_signed_shap",
        "ci_lo",
        "ci_hi",
        "feature_count",
    ):
        assert col in df.columns, f"missing {col}"
    assert len(df) == len(graph)
    assert df.attrs["ci"] == 0.95
    assert df.attrs["n_bootstrap"] == 50
    assert df.attrs["signed"] is True


def test_bootstrap_importance_ci_sandwiches_mean(graph, shap_arr) -> None:
    names, arr = shap_arr
    df = bootstrap_importance(graph, names, arr, n_bootstrap=200, random_state=0)
    rows = df[df["feature_count"] > 0]
    assert (rows["ci_lo"] <= rows["mean_signed_shap"] + 1e-9).all()
    assert (rows["mean_signed_shap"] <= rows["ci_hi"] + 1e-9).all()


def test_bootstrap_importance_unsigned_is_nonnegative(graph, shap_arr) -> None:
    names, arr = shap_arr
    df = bootstrap_importance(graph, names, arr, n_bootstrap=50, random_state=0, agg="mean_abs")
    assert "mean_abs_shap" in df.columns
    rows = df[df["feature_count"] > 0]
    assert (rows["mean_abs_shap"] >= 0).all()
    assert (rows["ci_lo"] >= 0).all()


def test_bootstrap_importance_deprecated_signed_still_works(graph, shap_arr) -> None:
    names, arr = shap_arr
    with pytest.warns(DeprecationWarning, match="signed"):
        df = bootstrap_importance(graph, names, arr, n_bootstrap=50, random_state=0, signed=False)
    assert "mean_abs_shap" in df.columns
    assert df.attrs["agg"] == "mean_abs"


def test_bootstrap_importance_rejects_unknown_agg(graph, shap_arr) -> None:
    names, arr = shap_arr
    with pytest.raises(ValueError, match="agg"):
        bootstrap_importance(graph, names, arr, agg="median")  # type: ignore[arg-type]


def test_bootstrap_importance_root_aggregates_all_features(graph, shap_arr) -> None:
    names, arr = shap_arr
    df = bootstrap_importance(graph, names, arr, n_bootstrap=50, random_state=0)
    root_row = df[df["name"] == graph.root].iloc[0]
    # Root sums every feature → its per-sample value equals the row sum of arr
    expected_mean = arr.sum(axis=1).mean()
    # Bootstrap mean should be close to the analytical mean (loose tolerance)
    assert abs(root_row["mean_signed_shap"] - expected_mean) < 0.05


def test_bootstrap_importance_rejects_bad_input(graph) -> None:
    with pytest.raises(ValueError, match="2D"):
        bootstrap_importance(graph, ["x1", "x2", "y1", "y2", "y3"], np.zeros(5))
    with pytest.raises(ValueError, match="cols"):
        bootstrap_importance(graph, ["x1", "x2"], np.zeros((10, 5)))
    with pytest.raises(ValueError, match="ci"):
        bootstrap_importance(graph, ["x1", "x2", "y1", "y2", "y3"], np.zeros((10, 5)), ci=1.5)
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_importance(
            graph, ["x1", "x2", "y1", "y2", "y3"], np.zeros((10, 5)), n_bootstrap=0
        )


def test_signed_concept_bar_renders(graph, shap_arr) -> None:
    names, arr = shap_arr
    df = bootstrap_importance(graph, names, arr, n_bootstrap=20, random_state=0)
    fig = signed_concept_bar(graph, df)
    assert fig.data
    assert fig.data[0].type == "bar"
    # only_concepts=True default → root + Income + Behaviour, minus root = 2 bars
    assert len(fig.data[0].y) == 2
    assert set(fig.data[0].y) == {"Income", "Behaviour"}


def test_signed_concept_bar_error_bars_match_ci(graph, shap_arr) -> None:
    names, arr = shap_arr
    df = bootstrap_importance(graph, names, arr, n_bootstrap=20, random_state=0)
    fig = signed_concept_bar(graph, df, sort=False)
    err_plus = np.asarray(fig.data[0].error_x.array, dtype=float)
    err_minus = np.asarray(fig.data[0].error_x.arrayminus, dtype=float)
    assert (err_plus >= -1e-9).all()
    assert (err_minus >= -1e-9).all()


def test_signed_concept_bar_sort_by_abs_mean(graph, shap_arr) -> None:
    names, arr = shap_arr
    df = bootstrap_importance(graph, names, arr, n_bootstrap=50, random_state=0)
    fig = signed_concept_bar(graph, df, sort=True)
    abs_means = np.abs(np.asarray(fig.data[0].x, dtype=float))
    assert (np.diff(abs_means) <= 1e-9).all(), "bars should be sorted by |mean| desc"


def test_signed_concept_bar_max_concepts_caps_output(graph, shap_arr) -> None:
    names, arr = shap_arr
    df = bootstrap_importance(graph, names, arr, n_bootstrap=20, random_state=0)
    fig = signed_concept_bar(graph, df, max_concepts=1)
    assert len(fig.data[0].y) == 1


def test_signed_concept_bar_branch_color_per_top_branch(graph, shap_arr) -> None:
    names, arr = shap_arr
    df = bootstrap_importance(graph, names, arr, n_bootstrap=20, random_state=0)
    fig = signed_concept_bar(graph, df, sort=False)
    colors = list(fig.data[0].marker.color)
    # Two top-level branches → two distinct colours
    assert len(set(colors)) == 2
