"""Smoke tests for the plotting layer (figure produced, PNG export viable)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from concept_graph_xai import (
    auc_drop,
    auc_drop_map,
    feature_counts,
    importance_sum,
    sunburst,
    utilization,
    utilization_map,
)


def test_sunburst_count_renders(graph) -> None:
    df = feature_counts(graph)
    fig = sunburst(graph, df, value="count", title="Counts")
    assert fig.data
    assert fig.data[0].type == "sunburst"
    assert len(fig.data[0].labels) == len(df)


def test_sunburst_importance_sum_with_colorscale(graph, toy) -> None:
    rng = np.random.default_rng(0)
    importances = rng.standard_normal(len(toy.feature_names))
    df = importance_sum(graph, toy.feature_names, importances)
    fig = sunburst(graph, df, value="importance_sum", colorscale="Viridis")
    assert fig.data[0].marker.colorscale is not None


def test_utilization_map_uses_used_color(graph, toy) -> None:
    rng = np.random.default_rng(0)
    importances = np.abs(rng.standard_normal(len(toy.feature_names)))
    importances[-1] = 0.0
    importances[-2] = 0.0
    df = utilization(graph, toy.feature_names, importances, threshold=0.0)
    fig = utilization_map(graph, df, used_color="#1f77b4", unused_color="#cccccc")
    colors = list(fig.data[0].marker.colors)
    assert "#cccccc" in colors


def test_auc_drop_map_renders(graph, fitted_model, toy) -> None:
    df = auc_drop(
        graph,
        fitted_model["model"],
        fitted_model["X_test"],
        fitted_model["y_test"],
        feature_names=toy.feature_names,
        strategy="permutation",
        n_repeats=2,
        random_state=0,
    )
    fig = auc_drop_map(graph, df)
    assert fig.data
    assert fig.data[0].type == "sunburst"
    values = list(fig.data[0].values)
    assert max(values) > 0, "all sector sizes are zero (chart will render empty)"
    root_idx = list(fig.data[0].labels).index(graph.root)
    children_total = sum(
        v
        for v, parent in zip(fig.data[0].values, fig.data[0].parents, strict=True)
        if parent == fig.data[0].ids[root_idx]
    )
    assert values[root_idx] >= children_total, (
        "root size must be >= sum of direct children for branchvalues='total'"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("kaleido") is None,
    reason="kaleido not installed; skipping PNG export check",
)
def test_png_export_runs(tmp_path, graph) -> None:
    df = feature_counts(graph)
    fig = sunburst(graph, df, value="count")
    out = tmp_path / "x.png"
    fig.write_image(out, width=600, height=600)
    assert out.exists()
    assert out.stat().st_size > 0
