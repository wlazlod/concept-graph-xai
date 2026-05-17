"""Tests for v0.6 concept_disparity + concept_disparity_heatmap (P11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from concept_graph_xai import (
    concept_disparity,
    concept_disparity_heatmap,
)


@pytest.fixture
def protected_series() -> pd.Series:
    return pd.Series(["A"] * 30 + ["B"] * 20 + ["C"] * 10)


def test_concept_disparity_long_form_columns(simple_graph, shap_arr, protected_series) -> None:
    names, arr = shap_arr
    df = concept_disparity(simple_graph, names, arr, protected_series, reference="A")
    for col in (
        "name",
        "kind",
        "depth",
        "parent",
        "path",
        "protected_group",
        "value",
        "reference_value",
        "feature_count",
    ):
        assert col in df.columns
    n_groups = 3
    assert len(df) == n_groups * len(simple_graph)
    assert df.attrs["agg"] == "mean_abs"
    assert df.attrs["reference_group"] == "A"
    assert df.attrs["protected_order"] == ["A", "B", "C"]


def test_concept_disparity_reference_row_is_zero(simple_graph, shap_arr, protected_series) -> None:
    names, arr = shap_arr
    df = concept_disparity(simple_graph, names, arr, protected_series, reference="A")
    ref_rows = df[df["protected_group"] == "A"]
    assert (ref_rows["value"] == 0).all()


def test_concept_disparity_gap_plus_reference_equals_group_value(
    simple_graph, shap_arr, protected_series
) -> None:
    names, arr = shap_arr
    df = concept_disparity(simple_graph, names, arr, protected_series, reference="A")
    # For group B, compute the analytic per-group value and confirm
    # value + reference_value = group_value.
    income_idxs = [names.index(f) for f in ("x1", "x2")]
    b_rows = (protected_series == "B").to_numpy()
    expected_income_b = float(np.abs(arr[b_rows][:, income_idxs].sum(axis=1)).mean())
    income_b = df.loc[
        (df["name"] == "Income") & (df["protected_group"] == "B"),
        ["value", "reference_value"],
    ].iloc[0]
    assert income_b["value"] + income_b["reference_value"] == pytest.approx(expected_income_b)


def test_concept_disparity_string_form_requires_X(simple_graph, shap_arr, protected_series) -> None:
    names, arr = shap_arr
    X = pd.DataFrame(arr, columns=names).assign(cohort=protected_series.to_numpy())
    df = concept_disparity(simple_graph, names, arr, "cohort", X=X, reference="A")
    assert set(df["protected_group"]) == {"A", "B", "C"}

    with pytest.raises(ValueError, match="X="):
        concept_disparity(simple_graph, names, arr, "cohort", reference="A")
    with pytest.raises(KeyError, match="cohort"):
        concept_disparity(
            simple_graph, names, arr, "cohort", X=X.drop(columns="cohort"), reference="A"
        )


def test_concept_disparity_unknown_reference_raises(
    simple_graph, shap_arr, protected_series
) -> None:
    names, arr = shap_arr
    with pytest.raises(KeyError, match="reference group"):
        concept_disparity(simple_graph, names, arr, protected_series, reference="Z")


def test_concept_disparity_mean_signed_allows_negative_gap(
    simple_graph, shap_arr, protected_series
) -> None:
    names, arr = shap_arr
    df = concept_disparity(
        simple_graph, names, arr, protected_series, reference="A", agg="mean_signed"
    )
    assert df.attrs["agg"] == "mean_signed"
    # Non-reference rows must include some negative gaps on random N(0,1) input.
    non_ref = df[df["protected_group"] != "A"]
    assert (non_ref["value"] < 0).any()


def test_concept_disparity_rejects_bad_input(simple_graph, shap_arr, protected_series) -> None:
    names, arr = shap_arr
    with pytest.raises(ValueError, match="2D"):
        concept_disparity(simple_graph, names, arr.sum(axis=1), protected_series, reference="A")
    with pytest.raises(ValueError, match="cols"):
        concept_disparity(simple_graph, names[:2], arr, protected_series, reference="A")
    with pytest.raises(ValueError, match=r"\d+ rows"):
        concept_disparity(simple_graph, names, arr, protected_series.iloc[:10], reference="A")


def test_concept_disparity_heatmap_renders(simple_graph, shap_arr, protected_series) -> None:
    names, arr = shap_arr
    df = concept_disparity(simple_graph, names, arr, protected_series, reference="A")
    fig = concept_disparity_heatmap(simple_graph, df)
    assert fig.data
    assert fig.data[0].type == "heatmap"
    # Reference column present by default; order is reference-first
    assert next(iter(fig.data[0].x)) == "A"
    # Reference column is exactly zero in every row
    z = np.asarray(fig.data[0].z, dtype=float)
    ref_col = list(fig.data[0].x).index("A")
    assert (z[:, ref_col] == 0).all()


def test_concept_disparity_heatmap_drops_reference_with_toggle(
    simple_graph, shap_arr, protected_series
) -> None:
    names, arr = shap_arr
    df = concept_disparity(simple_graph, names, arr, protected_series, reference="A")
    fig = concept_disparity_heatmap(simple_graph, df, include_reference=False)
    assert "A" not in list(fig.data[0].x)
    assert set(fig.data[0].x) == {"B", "C"}


def test_concept_disparity_heatmap_diverging_centred_at_zero(
    simple_graph, shap_arr, protected_series
) -> None:
    names, arr = shap_arr
    df = concept_disparity(simple_graph, names, arr, protected_series, reference="A")
    fig = concept_disparity_heatmap(simple_graph, df)
    assert fig.data[0].zmid == 0.0
    assert fig.data[0].zmin == -fig.data[0].zmax


def test_concept_disparity_heatmap_sort_by_max_abs(
    simple_graph, shap_arr, protected_series
) -> None:
    names, arr = shap_arr
    df = concept_disparity(simple_graph, names, arr, protected_series, reference="A")
    fig = concept_disparity_heatmap(simple_graph, df, sort_by="max_abs")
    z = np.asarray(fig.data[0].z, dtype=float)
    abs_max = np.abs(z).max(axis=1)
    assert (np.diff(abs_max) <= 1e-9).all(), "rows should be sorted by max(|gap|) desc"


def test_concept_disparity_heatmap_max_concepts_caps(
    simple_graph, shap_arr, protected_series
) -> None:
    names, arr = shap_arr
    df = concept_disparity(simple_graph, names, arr, protected_series, reference="A")
    fig = concept_disparity_heatmap(simple_graph, df, max_concepts=1)
    assert len(fig.data[0].y) == 1


def test_concept_disparity_heatmap_title_includes_reference(
    simple_graph, shap_arr, protected_series
) -> None:
    names, arr = shap_arr
    df = concept_disparity(simple_graph, names, arr, protected_series, reference="A")
    fig = concept_disparity_heatmap(simple_graph, df)
    assert "A" in fig.layout.title.text


def test_concept_disparity_heatmap_rejects_missing_columns(simple_graph) -> None:
    bad_df = pd.DataFrame({"name": ["a"], "kind": ["concept"], "value": [1.0]})
    with pytest.raises(KeyError, match="protected_group"):
        concept_disparity_heatmap(simple_graph, bad_df)
