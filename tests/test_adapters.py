"""Unit tests for the adapter layer."""

from __future__ import annotations

import numpy as np
import pytest

from concept_graph_xai.adapters import (
    from_feature_importances_,
    from_permutation_importance,
    from_shap_explanation,
)


class _FakeExplanation:
    def __init__(self, values: np.ndarray, feature_names: list[str]) -> None:
        self.values = values
        self.feature_names = feature_names


class _FakeBunch:
    def __init__(self, importances_mean: np.ndarray) -> None:
        self.importances_mean = importances_mean
        self.importances_std = np.zeros_like(importances_mean)


def test_from_shap_explanation_handles_2d_values() -> None:
    exp = _FakeExplanation(values=np.ones((10, 3)), feature_names=["a", "b", "c"])
    arr, names = from_shap_explanation(exp)
    assert arr.shape == (10, 3)
    assert names == ["a", "b", "c"]


def test_from_shap_explanation_picks_class_index_for_3d() -> None:
    values = np.zeros((5, 4, 2))
    values[..., 1] = 1.0
    exp = _FakeExplanation(values=values, feature_names=["a", "b", "c", "d"])
    arr, _ = from_shap_explanation(exp)
    assert np.allclose(arr, 1.0)


def test_from_shap_explanation_requires_names_when_missing() -> None:
    arr = np.zeros((4, 2))
    with pytest.raises(ValueError, match="feature_names"):
        from_shap_explanation(arr)


def test_from_permutation_importance_extracts_mean() -> None:
    bunch = _FakeBunch(np.array([0.1, 0.5, 0.0]))
    values, names = from_permutation_importance(bunch, ["a", "b", "c"])
    assert names == ["a", "b", "c"]
    assert np.allclose(values, [0.1, 0.5, 0.0])


def test_from_feature_importances_reads_attribute() -> None:
    class _Model:
        feature_importances_ = np.array([0.2, 0.8])

    values, names = from_feature_importances_(_Model(), ["a", "b"])
    assert names == ["a", "b"]
    assert np.allclose(values, [0.2, 0.8])
