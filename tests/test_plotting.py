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
    # hide_root=True by default → root sector dropped
    assert len(fig.data[0].labels) == len(df) - 1
    assert graph.root not in list(fig.data[0].labels)


def test_sunburst_with_root_shown(graph) -> None:
    df = feature_counts(graph)
    fig = sunburst(graph, df, value="count", hide_root=False)
    assert len(fig.data[0].labels) == len(df)
    assert graph.root in list(fig.data[0].labels)


def test_sunburst_importance_sum_with_colorscale(graph, toy) -> None:
    rng = np.random.default_rng(0)
    importances = rng.standard_normal(len(toy.feature_names))
    df = importance_sum(graph, toy.feature_names, importances)
    fig = sunburst(graph, df, value="importance_sum", colorscale="Viridis")
    assert fig.data[0].marker.colorscale is not None
    # marker.colors must carry one continuous value per sector and the
    # colorbar should be visible for a colorscale plot.
    sectors = len(fig.data[0].labels)
    assert sectors == len(fig.data[0].marker.colors)
    assert fig.data[0].marker.showscale is True


def test_sunburst_branch_colors_by_default(graph, toy) -> None:
    rng = np.random.default_rng(0)
    importances = np.abs(rng.standard_normal(len(toy.feature_names)))
    df = importance_sum(graph, toy.feature_names, importances)
    fig = sunburst(graph, df, value="importance_sum")
    # No colorscale → branch coloring kicks in by default
    assert fig.data[0].marker.colorscale is None
    colors = list(fig.data[0].marker.colors)
    assert colors, "expected per-sector colours from branch coloring"
    # Top-level branches should each have a distinct colour
    top_branches = [n for n in graph.children_of(graph.root)]
    assert len(set(colors)) >= min(len(top_branches), 2)


def test_sunburst_color_by_branch_inherits_to_descendants(graph, toy) -> None:
    rng = np.random.default_rng(0)
    importances = np.abs(rng.standard_normal(len(toy.feature_names)))
    df = importance_sum(graph, toy.feature_names, importances)
    fig = sunburst(graph, df, value="importance_sum", color_by="branch")
    labels = list(fig.data[0].labels)
    parents = list(fig.data[0].parents)
    colors = list(fig.data[0].marker.colors)

    # Hex → relative-luminance helper for "lighter" comparisons.
    def _lum(hex_color: str) -> float:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for branch in graph.children_of(graph.root):
        idx = labels.index(branch)
        branch_color = colors[idx]
        # branch is reparented to "" when hide_root=True
        assert parents[idx] == ""
        descendants = graph.descendants_of(branch)
        if not descendants:
            continue
        for child_label in descendants:
            if child_label not in labels:
                continue
            child_idx = labels.index(child_label)
            # Descendants share the branch hue but get progressively lighter.
            assert _lum(colors[child_idx]) >= _lum(branch_color) - 1e-6
        return
    pytest.skip("no multi-node branch in the toy graph")


def test_branch_colors_distinguish_top_level_branches(graph, toy) -> None:
    # Each top-level branch should get a different base hue.
    rng = np.random.default_rng(0)
    importances = np.abs(rng.standard_normal(len(toy.feature_names)))
    df = importance_sum(graph, toy.feature_names, importances)
    fig = sunburst(graph, df, value="importance_sum", color_by="branch")
    labels = list(fig.data[0].labels)
    colors = list(fig.data[0].marker.colors)
    branch_to_color = {b: colors[labels.index(b)] for b in graph.children_of(graph.root)}
    assert len(set(branch_to_color.values())) == len(branch_to_color)


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
    # hide_root=True by default → root not present, but the former direct
    # children of root must still be emitted with parent=""
    labels = list(fig.data[0].labels)
    parents = list(fig.data[0].parents)
    assert graph.root not in labels
    promoted = [labels[i] for i, p in enumerate(parents) if p == ""]
    assert set(promoted) == set(graph.children_of(graph.root))


def test_auc_drop_map_renders_with_root(graph, fitted_model, toy) -> None:
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
    fig = auc_drop_map(graph, df, hide_root=False)
    labels = list(fig.data[0].labels)
    assert graph.root in labels
    root_idx = labels.index(graph.root)
    children_total = sum(
        v
        for v, parent in zip(fig.data[0].values, fig.data[0].parents, strict=True)
        if parent == fig.data[0].ids[root_idx]
    )
    assert fig.data[0].values[root_idx] >= children_total, (
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
