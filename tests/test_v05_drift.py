"""Tests for v0.5 attribution_drift + concept_drift_lines + concept_drift_sunburst (P9, P10)."""

from __future__ import annotations

import numpy as np
import pytest

from concept_graph_xai import (
    attribution_drift,
    concept_drift_delta,
    concept_drift_lines,
    concept_drift_sunburst,
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


def test_attribution_drift_long_form_columns(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods)
    for col in ("name", "kind", "depth", "parent", "path", "period", "value", "feature_count"):
        assert col in df.columns
    n_periods = len(periods)
    assert len(df) == n_periods * len(simple_graph)
    assert df.attrs["agg"] == "mean_abs"


def test_attribution_drift_preserves_period_order(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods)
    assert df.attrs["period_order"] == ["Q1", "Q2", "Q3"]
    # First N rows belong to Q1, next N to Q2, etc.
    n = len(simple_graph)
    assert (df["period"].iloc[:n] == "Q1").all()
    assert (df["period"].iloc[n : 2 * n] == "Q2").all()


def test_attribution_drift_mean_abs_is_nonnegative(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods, agg="mean_abs")
    assert (df["value"] >= 0).all()


def test_attribution_drift_mean_signed_can_be_negative(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods, agg="mean_signed")
    assert df.attrs["agg"] == "mean_signed"
    # Some values must straddle zero on random N(0,1) input across periods
    assert (df["value"] > 0).any() and (df["value"] < 0).any()


def test_attribution_drift_matches_analytic(simple_graph, periods) -> None:
    _, q1_arr, _names = periods[0]
    df = attribution_drift(simple_graph, periods, agg="mean_abs")
    # Behaviour @ Q1: mean_n |y1+y2+y3| over Q1's 40 rows
    expected = float(np.abs(q1_arr[:, 2:].sum(axis=1)).mean())
    behav_q1 = df.loc[(df["name"] == "Behaviour") & (df["period"] == "Q1"), "value"].iloc[0]
    assert behav_q1 == pytest.approx(expected)


def test_attribution_drift_rejects_bad_input(simple_graph, periods) -> None:
    with pytest.raises(ValueError, match="at least one"):
        attribution_drift(simple_graph, [])
    bad = [(periods[0][0], periods[0][1].sum(axis=1), periods[0][2])]
    with pytest.raises(ValueError, match="2D"):
        attribution_drift(simple_graph, bad)
    bad2 = [(periods[0][0], periods[0][1], ["x1", "x2"])]
    with pytest.raises(ValueError, match="cols"):
        attribution_drift(simple_graph, bad2)
    # Duplicate period labels rejected
    dup = [periods[0], (periods[0][0], periods[1][1], periods[1][2])]
    with pytest.raises(ValueError, match="unique"):
        attribution_drift(simple_graph, dup)


def test_concept_drift_lines_renders(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods)
    fig = concept_drift_lines(simple_graph, df)
    assert fig.data
    for trace in fig.data:
        assert trace.type == "scatter"


def test_concept_drift_lines_one_trace_per_concept(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods)
    fig = concept_drift_lines(simple_graph, df, max_concepts=None)
    # only_concepts=True default + root dropped -> 2 concepts in this simple_graph
    assert len(fig.data) == 2
    assert {tr.name for tr in fig.data} == {"Income", "Behaviour"}


def test_concept_drift_lines_max_concepts_caps(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods)
    fig = concept_drift_lines(simple_graph, df, max_concepts=1)
    assert len(fig.data) == 1
    # The title should mention the cap honestly: "top K of N concepts".
    assert "top 1 of 2 concepts" in fig.layout.title.text


def test_concept_drift_lines_title_omits_cap_when_no_filter(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods)
    fig = concept_drift_lines(simple_graph, df, max_concepts=None)
    assert "top" not in fig.layout.title.text


def test_concept_drift_lines_x_axis_matches_period_order(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods)
    fig = concept_drift_lines(simple_graph, df)
    for trace in fig.data:
        assert list(trace.x) == ["Q1", "Q2", "Q3"]


def test_concept_drift_lines_deprecated_top_k_still_works(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods)
    with pytest.warns(DeprecationWarning, match="top_k"):
        fig = concept_drift_lines(simple_graph, df, top_k=1)
    assert len(fig.data) == 1


def test_concept_drift_lines_deprecated_include_root_still_works(simple_graph, periods) -> None:
    df = attribution_drift(simple_graph, periods)
    with pytest.warns(DeprecationWarning, match="include_root"):
        fig = concept_drift_lines(simple_graph, df, include_root=True)
    # include_root=True -> hide_root=False -> root concept is now present in the
    # legend on top of the regular concepts.
    names = {tr.name for tr in fig.data}
    assert simple_graph.root in names


def test_concept_drift_lines_rejects_missing_columns(simple_graph) -> None:
    import pandas as pd

    bad_df = pd.DataFrame({"name": ["a"], "value": [1.0]})
    with pytest.raises(KeyError, match=r"kind|period"):
        concept_drift_lines(simple_graph, bad_df)


# ---------------------------------------------------------------------------
# H.2 — concept_drift_delta + concept_drift_sunburst
# ---------------------------------------------------------------------------


def test_concept_drift_delta_default_baseline_target(simple_graph, periods) -> None:
    delta_df = concept_drift_delta(simple_graph, periods)
    assert delta_df.attrs["baseline_period"] == "Q1"
    assert delta_df.attrs["target_period"] == "Q3"
    for col in ("name", "kind", "depth", "parent", "baseline", "target", "delta", "feature_count"):
        assert col in delta_df.columns
    assert len(delta_df) == len(simple_graph)


def test_concept_drift_delta_equals_target_minus_baseline(simple_graph, periods) -> None:
    delta_df = concept_drift_delta(simple_graph, periods, baseline="Q1", target="Q3")
    np.testing.assert_allclose(
        delta_df["delta"].to_numpy(dtype=float),
        delta_df["target"].to_numpy(dtype=float) - delta_df["baseline"].to_numpy(dtype=float),
        atol=1e-12,
    )


def test_concept_drift_delta_explicit_period_selection(simple_graph, periods) -> None:
    delta_df = concept_drift_delta(simple_graph, periods, baseline="Q2", target="Q1")
    assert delta_df.attrs["baseline_period"] == "Q2"
    assert delta_df.attrs["target_period"] == "Q1"


def test_concept_drift_delta_rejects_bad_input(simple_graph, periods) -> None:
    with pytest.raises(ValueError, match="at least one"):
        concept_drift_delta(simple_graph, [])
    with pytest.raises(KeyError, match="baseline period"):
        concept_drift_delta(simple_graph, periods, baseline="nope")
    with pytest.raises(KeyError, match="target period"):
        concept_drift_delta(simple_graph, periods, target="nope")
    with pytest.raises(ValueError, match="must differ"):
        concept_drift_delta(simple_graph, periods, baseline="Q1", target="Q1")


def test_concept_drift_sunburst_renders(simple_graph, periods) -> None:
    delta_df = concept_drift_delta(simple_graph, periods)
    fig = concept_drift_sunburst(simple_graph, delta_df)
    assert fig.data
    trace = fig.data[0]
    assert trace.type == "sunburst"
    # hide_root=True default → root absent
    assert simple_graph.root not in list(trace.labels)
    # Diverging colorscale centred at 0
    assert trace.marker.cmid == 0.0
    assert trace.marker.cmin == -trace.marker.cmax


def test_concept_drift_sunburst_uses_baseline_target_in_title(simple_graph, periods) -> None:
    delta_df = concept_drift_delta(simple_graph, periods, baseline="Q1", target="Q3")
    fig = concept_drift_sunburst(simple_graph, delta_df)
    assert "Q1" in fig.layout.title.text
    assert "Q3" in fig.layout.title.text


def test_concept_drift_sunburst_with_root_shown(simple_graph, periods) -> None:
    delta_df = concept_drift_delta(simple_graph, periods)
    fig = concept_drift_sunburst(simple_graph, delta_df, hide_root=False)
    assert simple_graph.root in list(fig.data[0].labels)


def test_concept_drift_sunburst_rejects_missing_columns(simple_graph, periods) -> None:
    import pandas as pd

    delta_df = concept_drift_delta(simple_graph, periods).drop(columns=["delta"])
    with pytest.raises(KeyError, match="delta"):
        concept_drift_sunburst(simple_graph, delta_df)

    delta_df2 = concept_drift_delta(simple_graph, periods).drop(columns=["feature_count"])
    with pytest.raises(KeyError, match="feature_count"):
        concept_drift_sunburst(simple_graph, delta_df2)
    _ = pd  # silence unused if pytest skips


def test_concept_drift_sunburst_section_sizes_match_feature_count(simple_graph, periods) -> None:
    delta_df = concept_drift_delta(simple_graph, periods)
    fig = concept_drift_sunburst(simple_graph, delta_df)
    labels = list(fig.data[0].labels)
    values = list(fig.data[0].values)
    label_to_value = dict(zip(labels, values, strict=True))
    # Income has 2 features (x1, x2); Behaviour has 3
    assert label_to_value["Income"] == 2
    assert label_to_value["Behaviour"] == 3
