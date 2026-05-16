"""Tests for v0.5 concept_pareto (P8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from concept_graph_xai import concept_pareto, segment_importance


def test_concept_pareto_one_trace_per_segment_plus_equality(
    simple_graph, shap_arr, segments_series
) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series)
    fig = concept_pareto(simple_graph, df)
    # 1 equality line + 3 cohort lines
    assert len(fig.data) == 1 + len(set(segments_series))
    names_in_legend = [tr.name for tr in fig.data]
    assert "equality" in names_in_legend
    for seg in set(segments_series):
        assert seg in names_in_legend


def test_concept_pareto_curves_are_monotonic_and_end_at_one(
    simple_graph, shap_arr, segments_series
) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series)
    fig = concept_pareto(simple_graph, df)
    for trace in fig.data:
        if trace.name == "equality":
            continue
        ys = np.asarray(trace.y, dtype=float)
        xs = np.asarray(trace.x, dtype=float)
        assert ys[0] == pytest.approx(0.0)
        assert xs[0] == pytest.approx(0.0)
        assert ys[-1] == pytest.approx(1.0, abs=1e-9)
        assert xs[-1] == pytest.approx(1.0, abs=1e-9)
        # Non-decreasing
        assert (np.diff(ys) >= -1e-9).all()
        assert (np.diff(xs) >= -1e-9).all()


def test_concept_pareto_respects_segment_order(simple_graph, shap_arr) -> None:
    names, arr = shap_arr
    series = pd.Series(
        pd.Categorical(
            np.tile(["low", "mid", "high"], 20),
            categories=["low", "mid", "high"],
            ordered=True,
        )
    )
    df = segment_importance(simple_graph, names, arr, series)
    fig = concept_pareto(simple_graph, df)
    cohort_traces = [tr for tr in fig.data if tr.name != "equality"]
    assert [tr.name for tr in cohort_traces] == ["low", "mid", "high"]


def test_concept_pareto_equality_line_toggle(simple_graph, shap_arr, segments_series) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series)
    fig = concept_pareto(simple_graph, df, show_equality_line=False)
    assert all(tr.name != "equality" for tr in fig.data)
    assert len(fig.data) == len(set(segments_series))


def test_concept_pareto_skips_zero_importance_segments(simple_graph, shap_arr) -> None:
    # Two real cohorts + one with all zeros. The zero cohort should be skipped.
    names, arr = shap_arr
    series = pd.Series(["A"] * 20 + ["B"] * 20 + ["zero"] * 20)
    df = segment_importance(simple_graph, names, arr, series)
    # zero out the "zero" segment's value rows
    df.loc[df["segment"] == "zero", "value"] = 0.0
    df.attrs["segment_order"] = ["A", "B", "zero"]
    fig = concept_pareto(simple_graph, df)
    cohort_names = [tr.name for tr in fig.data if tr.name != "equality"]
    assert "zero" not in cohort_names
    assert {"A", "B"}.issubset(set(cohort_names))


def test_concept_pareto_rejects_empty_or_all_zero(simple_graph, shap_arr, segments_series) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series)
    with pytest.raises(ValueError, match="empty"):
        concept_pareto(simple_graph, df.iloc[0:0])
    # All segments zero -> nothing to plot
    df_all_zero = df.copy()
    df_all_zero["value"] = 0.0
    with pytest.raises(ValueError, match="non-zero"):
        concept_pareto(simple_graph, df_all_zero)


def test_concept_pareto_drops_root_by_default(simple_graph, shap_arr, segments_series) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series)
    fig = concept_pareto(simple_graph, df)
    for trace in fig.data:
        if trace.name == "equality":
            continue
        # The first hover label is the placeholder "—", then the ranked concept
        # names. Root must not appear.
        assert simple_graph.root not in list(trace.text)


def test_concept_pareto_includes_root_when_asked(simple_graph, shap_arr, segments_series) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series)
    fig = concept_pareto(simple_graph, df, hide_root=False)
    found = False
    for trace in fig.data:
        if trace.name == "equality":
            continue
        if simple_graph.root in list(trace.text):
            found = True
            break
    assert found


def test_concept_pareto_rejects_missing_columns(simple_graph) -> None:
    bad_df = pd.DataFrame({"name": ["a"], "kind": ["concept"], "value": [1.0]})
    with pytest.raises(KeyError, match="segment"):
        concept_pareto(simple_graph, bad_df)


def test_concept_pareto_deprecated_include_root_still_works(
    simple_graph, shap_arr, segments_series
) -> None:
    names, arr = shap_arr
    df = segment_importance(simple_graph, names, arr, segments_series)
    with pytest.warns(DeprecationWarning, match="include_root"):
        fig = concept_pareto(simple_graph, df, include_root=True)
    # include_root=True -> hide_root=False -> Risk (the root) is now in the
    # ranked concept list and appears in at least one hover trace.
    cohort_traces = [tr for tr in fig.data if tr.name != "equality"]
    seen_root = False
    for trace in cohort_traces:
        if simple_graph.root in list(trace.text):
            seen_root = True
            break
    assert seen_root
