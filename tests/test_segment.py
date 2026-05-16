"""Tests for v0.5 segment_importance + segment_concept_heatmap (P7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from concept_graph_xai import (
    segment_concept_heatmap,
    segment_importance,
)


def test_segment_importance_long_form_columns(simple_graph, shap_arr, segments_series) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series)
    for col in ("name", "kind", "depth", "parent", "segment", "value", "feature_count"):
        assert col in df.columns
    n_segments = 3
    assert len(df) == n_segments * len(simple_graph)


def test_segment_importance_string_form_requires_X(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    X = pd.DataFrame(arr, columns=names).assign(cohort=np.tile(["young", "old"], 30))
    df = segment_importance(simple_graph, names, arr, "cohort", X=X)
    assert set(df["segment"]) == {"young", "old"}
    # string without X -> error
    with pytest.raises(ValueError, match="X="):
        segment_importance(simple_graph, names, arr, "cohort")
    # string with X but column missing
    with pytest.raises(KeyError, match="cohort"):
        segment_importance(simple_graph, names, arr, "cohort", X=X.drop(columns="cohort"))


def test_segment_importance_categorical_ordered_segments(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    series = pd.Series(
        pd.Categorical(
            np.tile(["low", "mid", "high"], 20),
            categories=["low", "mid", "high"],
            ordered=True,
        )
    )
    df = segment_importance(simple_graph, names, arr, series)
    # Categorical ordering should be preserved in attrs["segment_order"]
    assert df.attrs["segment_order"] == ["low", "mid", "high"]


def test_segment_importance_mean_abs_is_nonnegative(
    simple_graph, shap_arr, segments_series
) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series, agg="mean_abs")
    assert (df["value"] >= 0).all()


def test_segment_importance_mean_signed_can_be_negative(
    simple_graph, shap_arr, segments_series
) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series, agg="mean_signed")
    assert df.attrs["agg"] == "mean_signed"
    # With random N(0,1) SHAP and three segments, signed mean should hit both signs.
    assert (df["value"] > 0).any() and (df["value"] < 0).any()


def test_segment_importance_matches_analytic(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    series = pd.Series(["A"] * 30 + ["B"] * 30)
    df = segment_importance(simple_graph, names, arr, series, agg="mean_abs")
    # Income concept under "A": mean_n |x1+x2| over the first 30 rows
    expected_income_A = float(np.abs(arr[:30, :2].sum(axis=1)).mean())
    income_A = df.loc[(df["name"] == "Income") & (df["segment"] == "A"), "value"].iloc[0]
    assert income_A == pytest.approx(expected_income_A)


def test_segment_importance_rejects_bad_input(simple_graph, shap_arr, segments_series) -> None:
    names, arr = shap_arr
    with pytest.raises(ValueError, match="2D"):
        segment_importance(simple_graph, names, arr.sum(axis=1), segments_series)
    with pytest.raises(ValueError, match="cols"):
        segment_importance(simple_graph, names[:2], arr, segments_series)
    with pytest.raises(ValueError, match=r"\d+ rows"):
        segment_importance(simple_graph, names, arr, segments_series.iloc[:10])
    with pytest.raises(TypeError, match="Series or column-name"):
        segment_importance(simple_graph, names, arr, segments=42)  # type: ignore[arg-type]


def test_segment_concept_heatmap_renders(simple_graph, shap_arr, segments_series) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series)
    fig = segment_concept_heatmap(simple_graph, df)
    assert fig.data
    assert fig.data[0].type == "heatmap"
    # Root should be hidden by default; only Income and Behaviour remain
    y_labels = list(fig.data[0].y)
    assert simple_graph.root not in y_labels
    assert {"Income", "Behaviour"}.issubset(set(y_labels))


def test_segment_concept_heatmap_segment_order_preserved(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    series = pd.Series(
        pd.Categorical(
            np.tile(["low", "mid", "high"], 20),
            categories=["low", "mid", "high"],
            ordered=True,
        )
    )
    df = segment_importance(simple_graph, names, arr, series)
    fig = segment_concept_heatmap(simple_graph, df)
    assert list(fig.data[0].x) == ["low", "mid", "high"]


def test_segment_concept_heatmap_signed_uses_diverging(
    simple_graph, shap_arr, segments_series
) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series, agg="mean_signed")
    fig = segment_concept_heatmap(simple_graph, df)
    assert fig.data[0].zmid == 0.0


def test_segment_concept_heatmap_max_concepts_caps(simple_graph, shap_arr, segments_series) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series)
    fig = segment_concept_heatmap(simple_graph, df, max_concepts=1)
    assert len(fig.data[0].y) == 1
