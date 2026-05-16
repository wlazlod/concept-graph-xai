"""Tests for v0.5 attribution_drift + concept_drift_lines + concept_drift_sunburst (P9, P10)."""

from __future__ import annotations

import numpy as np
import pytest

from concept_graph_xai import (
    ConceptGraph,
    attribution_drift,
    concept_drift_delta,
    concept_drift_lines,
    concept_drift_sunburst,
)


@pytest.fixture
def graph() -> ConceptGraph:
    return ConceptGraph.from_dict(
        {"Risk": {"Income": ["x1", "x2"], "Behaviour": ["y1", "y2", "y3"]}}
    )


@pytest.fixture
def periods() -> list[tuple[str, np.ndarray, list[str]]]:
    rng = np.random.default_rng(0)
    names = ["x1", "x2", "y1", "y2", "y3"]
    return [
        ("Q1", rng.standard_normal((40, 5)), names),
        ("Q2", rng.standard_normal((50, 5)) * 1.5, names),
        ("Q3", rng.standard_normal((35, 5)) * 0.5, names),
    ]


def test_attribution_drift_long_form_columns(graph, periods) -> None:
    df = attribution_drift(graph, periods)
    for col in ("name", "kind", "depth", "parent", "path", "period", "value", "feature_count"):
        assert col in df.columns
    n_periods = len(periods)
    assert len(df) == n_periods * len(graph)
    assert df.attrs["agg"] == "mean_abs"


def test_attribution_drift_preserves_period_order(graph, periods) -> None:
    df = attribution_drift(graph, periods)
    assert df.attrs["period_order"] == ["Q1", "Q2", "Q3"]
    # First N rows belong to Q1, next N to Q2, etc.
    n = len(graph)
    assert (df["period"].iloc[:n] == "Q1").all()
    assert (df["period"].iloc[n : 2 * n] == "Q2").all()


def test_attribution_drift_mean_abs_is_nonnegative(graph, periods) -> None:
    df = attribution_drift(graph, periods, agg="mean_abs")
    assert (df["value"] >= 0).all()


def test_attribution_drift_mean_signed_can_be_negative(graph, periods) -> None:
    df = attribution_drift(graph, periods, agg="mean_signed")
    assert df.attrs["agg"] == "mean_signed"
    # Some values must straddle zero on random N(0,1) input across periods
    assert (df["value"] > 0).any() and (df["value"] < 0).any()


def test_attribution_drift_matches_analytic(graph, periods) -> None:
    _, q1_arr, _names = periods[0]
    df = attribution_drift(graph, periods, agg="mean_abs")
    # Behaviour @ Q1: mean_n |y1+y2+y3| over Q1's 40 rows
    expected = float(np.abs(q1_arr[:, 2:].sum(axis=1)).mean())
    behav_q1 = df.loc[
        (df["name"] == "Behaviour") & (df["period"] == "Q1"), "value"
    ].iloc[0]
    assert behav_q1 == pytest.approx(expected)


def test_attribution_drift_rejects_bad_input(graph, periods) -> None:
    with pytest.raises(ValueError, match="at least one"):
        attribution_drift(graph, [])
    bad = [(periods[0][0], periods[0][1].sum(axis=1), periods[0][2])]
    with pytest.raises(ValueError, match="2D"):
        attribution_drift(graph, bad)
    bad2 = [(periods[0][0], periods[0][1], ["x1", "x2"])]
    with pytest.raises(ValueError, match="cols"):
        attribution_drift(graph, bad2)
    # Duplicate period labels rejected
    dup = [periods[0], (periods[0][0], periods[1][1], periods[1][2])]
    with pytest.raises(ValueError, match="unique"):
        attribution_drift(graph, dup)


def test_concept_drift_lines_renders(graph, periods) -> None:
    df = attribution_drift(graph, periods)
    fig = concept_drift_lines(graph, df)
    assert fig.data
    for trace in fig.data:
        assert trace.type == "scatter"


def test_concept_drift_lines_one_trace_per_concept(graph, periods) -> None:
    df = attribution_drift(graph, periods)
    fig = concept_drift_lines(graph, df, top_k=None)
    # only_concepts=True default + root dropped -> 2 concepts in this graph
    assert len(fig.data) == 2
    assert {tr.name for tr in fig.data} == {"Income", "Behaviour"}


def test_concept_drift_lines_top_k_caps(graph, periods) -> None:
    df = attribution_drift(graph, periods)
    fig = concept_drift_lines(graph, df, top_k=1)
    assert len(fig.data) == 1


def test_concept_drift_lines_x_axis_matches_period_order(graph, periods) -> None:
    df = attribution_drift(graph, periods)
    fig = concept_drift_lines(graph, df)
    for trace in fig.data:
        assert list(trace.x) == ["Q1", "Q2", "Q3"]


def test_concept_drift_lines_rejects_missing_columns(graph) -> None:
    import pandas as pd

    bad_df = pd.DataFrame({"name": ["a"], "value": [1.0]})
    with pytest.raises(KeyError, match=r"kind|period"):
        concept_drift_lines(graph, bad_df)


# ---------------------------------------------------------------------------
# H.2 — concept_drift_delta + concept_drift_sunburst
# ---------------------------------------------------------------------------


def test_concept_drift_delta_default_baseline_target(graph, periods) -> None:
    delta_df = concept_drift_delta(graph, periods)
    assert delta_df.attrs["baseline_period"] == "Q1"
    assert delta_df.attrs["target_period"] == "Q3"
    for col in ("name", "kind", "depth", "parent", "baseline", "target", "delta", "feature_count"):
        assert col in delta_df.columns
    assert len(delta_df) == len(graph)


def test_concept_drift_delta_equals_target_minus_baseline(graph, periods) -> None:
    delta_df = concept_drift_delta(graph, periods, baseline="Q1", target="Q3")
    np.testing.assert_allclose(
        delta_df["delta"].to_numpy(dtype=float),
        delta_df["target"].to_numpy(dtype=float) - delta_df["baseline"].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_concept_drift_delta_explicit_period_selection(graph, periods) -> None:
    delta_df = concept_drift_delta(graph, periods, baseline="Q2", target="Q1")
    assert delta_df.attrs["baseline_period"] == "Q2"
    assert delta_df.attrs["target_period"] == "Q1"


def test_concept_drift_delta_rejects_bad_input(graph, periods) -> None:
    with pytest.raises(ValueError, match="at least one"):
        concept_drift_delta(graph, [])
    with pytest.raises(KeyError, match="baseline period"):
        concept_drift_delta(graph, periods, baseline="nope")
    with pytest.raises(KeyError, match="target period"):
        concept_drift_delta(graph, periods, target="nope")
    with pytest.raises(ValueError, match="must differ"):
        concept_drift_delta(graph, periods, baseline="Q1", target="Q1")


def test_concept_drift_sunburst_renders(graph, periods) -> None:
    delta_df = concept_drift_delta(graph, periods)
    fig = concept_drift_sunburst(graph, delta_df)
    assert fig.data
    trace = fig.data[0]
    assert trace.type == "sunburst"
    # hide_root=True default → root absent
    assert graph.root not in list(trace.labels)
    # Diverging colorscale centred at 0
    assert trace.marker.cmid == 0.0
    assert trace.marker.cmin == -trace.marker.cmax


def test_concept_drift_sunburst_uses_baseline_target_in_title(graph, periods) -> None:
    delta_df = concept_drift_delta(graph, periods, baseline="Q1", target="Q3")
    fig = concept_drift_sunburst(graph, delta_df)
    assert "Q1" in fig.layout.title.text
    assert "Q3" in fig.layout.title.text


def test_concept_drift_sunburst_with_root_shown(graph, periods) -> None:
    delta_df = concept_drift_delta(graph, periods)
    fig = concept_drift_sunburst(graph, delta_df, hide_root=False)
    assert graph.root in list(fig.data[0].labels)


def test_concept_drift_sunburst_rejects_missing_columns(graph, periods) -> None:
    import pandas as pd

    delta_df = concept_drift_delta(graph, periods).drop(columns=["delta"])
    with pytest.raises(KeyError, match="delta"):
        concept_drift_sunburst(graph, delta_df)

    delta_df2 = concept_drift_delta(graph, periods).drop(columns=["feature_count"])
    with pytest.raises(KeyError, match="feature_count"):
        concept_drift_sunburst(graph, delta_df2)
    _ = pd  # silence unused if pytest skips


def test_concept_drift_sunburst_section_sizes_match_feature_count(graph, periods) -> None:
    delta_df = concept_drift_delta(graph, periods)
    fig = concept_drift_sunburst(graph, delta_df)
    labels = list(fig.data[0].labels)
    values = list(fig.data[0].values)
    label_to_value = dict(zip(labels, values, strict=True))
    # Income has 2 features (x1, x2); Behaviour has 3
    assert label_to_value["Income"] == 2
    assert label_to_value["Behaviour"] == 3
