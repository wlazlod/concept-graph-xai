"""Edge-case smoke tests across the metric + plot families (T1, T4).

Parametrises every public SHAP-aggregation metric over the unhappy paths
that haven't otherwise been covered (empty input, single sample, all-zero
SHAP, single-feature concept). The goal is to catch crashes — exact
numeric output for these cases is not asserted, just that the call
returns a well-formed result or raises a clear error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from concept_graph_xai import (
    ConceptGraph,
    attribution_drift,
    bootstrap_importance,
    concept_disparity,
    concept_interaction_matrix,
    feature_correlation,
    importance_sum,
    segment_importance,
    shap_correlation,
    utilization,
)


@pytest.fixture
def graph() -> ConceptGraph:
    return ConceptGraph.from_dict(
        {"Risk": {"Income": ["x1", "x2"], "Behaviour": ["y1", "y2", "y3"]}}
    )


@pytest.fixture
def single_feature_graph() -> ConceptGraph:
    return ConceptGraph.from_dict({"Risk": {"Only": ["solo"]}})


def _names_for(graph: ConceptGraph) -> list[str]:
    return graph.features()


# ---------------------------------------------------------------------------
# T1 — edge cases for SHAP-aggregation metrics
# ---------------------------------------------------------------------------


def test_importance_sum_handles_empty_input(graph) -> None:
    names = _names_for(graph)
    arr = np.zeros((0, len(names)), dtype=float)
    # numpy emits a RuntimeWarning for mean-of-empty; that's expected here.
    with pytest.warns(RuntimeWarning):
        df = importance_sum(graph, names, arr)
    assert len(df) == len(graph)
    # Empty input → per-feature mean is NaN → per-concept sum is NaN.
    assert df["importance_sum"].isna().all()


def test_utilization_handles_empty_input(graph) -> None:
    names = _names_for(graph)
    arr = np.zeros((0, len(names)), dtype=float)
    with pytest.warns(RuntimeWarning):
        df = utilization(graph, names, arr, threshold=0.0)
    # NaN importances → no feature exceeds the threshold → is_used False.
    assert (~df["is_used"]).all()


def test_bootstrap_importance_handles_single_sample(graph) -> None:
    names = _names_for(graph)
    arr = np.array([[0.1, -0.2, 0.3, 0.0, -0.4]])
    df = bootstrap_importance(graph, names, arr, n_bootstrap=20, random_state=0)
    # With N=1 every resample picks the same row, so CI collapses to the
    # point estimate. ci_lo == ci_hi == mean for every concept.
    rows = df[df["feature_count"] > 0]
    np.testing.assert_allclose(
        rows["ci_lo"].to_numpy(), rows["mean_signed_shap"].to_numpy(), atol=1e-12
    )
    np.testing.assert_allclose(
        rows["ci_hi"].to_numpy(), rows["mean_signed_shap"].to_numpy(), atol=1e-12
    )


def test_segment_importance_handles_all_nan_segment(graph) -> None:
    names = _names_for(graph)
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((10, len(names)))
    series = pd.Series(["A", "A", "A", None, None, "B", "B", "B", None, None])
    df = segment_importance(graph, names, arr, series)
    # NA rows are dropped, B and A both appear, no extra "nan" group.
    assert set(df["segment"]) == {"A", "B"}


def test_concept_disparity_handles_single_feature_graph(single_feature_graph) -> None:
    names = ["solo"]
    arr = np.array([[0.5], [-0.1], [0.3], [0.0]])
    series = pd.Series(["A", "A", "B", "B"])
    df = concept_disparity(single_feature_graph, names, arr, series, reference="A")
    # Reference group's row is zero; B row is the difference.
    ref_rows = df[df["protected_group"] == "A"]
    assert (ref_rows["value"] == 0).all()


def test_attribution_drift_handles_single_period(graph) -> None:
    names = _names_for(graph)
    rng = np.random.default_rng(0)
    period = ("Q1", rng.standard_normal((20, len(names))), names)
    df = attribution_drift(graph, [period])
    assert df.attrs["period_order"] == ["Q1"]
    assert len(df) == len(graph)


def test_feature_correlation_handles_single_feature_input() -> None:
    # Graph has only one feature; correlation should not crash even though
    # corr() on a single column gives a 1x1 matrix.
    graph = ConceptGraph.from_dict({"Risk": {"Only": ["solo"]}})
    X = pd.DataFrame({"solo": [1.0, 2.0, 3.0, 4.0]})
    result = feature_correlation(graph, X)
    assert result.matrix.shape == (1, 1)


def test_concept_interaction_matrix_handles_single_concept_graph() -> None:
    graph = ConceptGraph.from_dict({"Risk": {"Only": ["solo"]}})
    rng = np.random.default_rng(0)
    arr = 0.5 * (
        rng.standard_normal((10, 1, 1)) + rng.standard_normal((10, 1, 1)).transpose(0, 2, 1)
    )
    df = concept_interaction_matrix(graph, ["solo"], arr)
    assert df.shape == (1, 1)


def test_bootstrap_importance_with_all_zero_shap(graph) -> None:
    names = _names_for(graph)
    arr = np.zeros((30, len(names)), dtype=float)
    df = bootstrap_importance(graph, names, arr, n_bootstrap=20, random_state=0)
    # Everything zero → mean 0, CI collapses to 0.
    rows = df[df["feature_count"] > 0]
    assert (rows["mean_signed_shap"] == 0).all()
    assert (rows["ci_lo"] == 0).all()
    assert (rows["ci_hi"] == 0).all()


# ---------------------------------------------------------------------------
# T4 — feature_correlation now accepts both DataFrame and ndarray
# ---------------------------------------------------------------------------


def test_feature_correlation_accepts_ndarray() -> None:
    graph = ConceptGraph.from_dict({"Risk": {"Income": ["x1", "x2"]}})
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((30, 2))
    result = feature_correlation(graph, arr, feature_names=["x1", "x2"])
    assert result.matrix.shape == (2, 2)
    # Should agree numerically with the DataFrame form.
    df = pd.DataFrame(arr, columns=["x1", "x2"])
    result_df = feature_correlation(graph, df)
    np.testing.assert_allclose(result.matrix.to_numpy(), result_df.matrix.to_numpy())


def test_feature_correlation_rejects_ndarray_without_feature_names() -> None:
    graph = ConceptGraph.from_dict({"Risk": {"Income": ["x1", "x2"]}})
    arr = np.zeros((10, 2))
    with pytest.raises(ValueError, match="feature_names"):
        feature_correlation(graph, arr)


def test_feature_correlation_rejects_ndarray_with_wrong_feature_count() -> None:
    graph = ConceptGraph.from_dict({"Risk": {"Income": ["x1", "x2"]}})
    arr = np.zeros((10, 3))
    with pytest.raises(ValueError, match="3 columns"):
        feature_correlation(graph, arr, feature_names=["x1", "x2"])


def test_shap_correlation_and_feature_correlation_agree_on_ndarray() -> None:
    # The two correlation entry points should accept the same array shape.
    graph = ConceptGraph.from_dict({"Risk": {"Income": ["x1", "x2"]}})
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((30, 2))
    a = feature_correlation(graph, arr, feature_names=["x1", "x2"])
    b = shap_correlation(graph, ["x1", "x2"], arr)
    # Both compute Spearman on the same data → identical matrices.
    np.testing.assert_allclose(a.matrix.to_numpy(), b.matrix.to_numpy())
