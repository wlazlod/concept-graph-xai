"""Adapter from a SHAP ``Explanation`` object to canonical arrays."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def from_shap_explanation(
    explanation: Any,
    *,
    feature_names: Sequence[str] | None = None,
    class_index: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert a SHAP ``Explanation`` (or compatible object) to ``(values, names)``.

    Parameters
    ----------
    explanation:
        Either a ``shap.Explanation`` instance or any object with ``.values`` and
        ``.feature_names`` attributes. A raw ``numpy.ndarray`` is also accepted
        when ``feature_names`` is provided.
    feature_names:
        Required only when ``explanation`` does not carry ``feature_names`` of
        its own.
    class_index:
        For multi-class explanations (3D ``values`` of shape ``(N, F, C)``),
        select one class. Defaults to the last class.
    """

    values = getattr(explanation, "values", explanation)
    arr = np.asarray(values, dtype=float)

    names: list[str]
    explanation_names = getattr(explanation, "feature_names", None)
    if feature_names is not None:
        names = list(feature_names)
    elif explanation_names is not None:
        names = list(explanation_names)
    else:
        raise ValueError("feature_names must be provided when the explanation has none")

    if arr.ndim == 3:
        idx = class_index if class_index is not None else arr.shape[2] - 1
        arr = arr[:, :, idx]

    if arr.ndim not in (1, 2):
        raise ValueError(f"unexpected SHAP values rank: {arr.ndim}")
    if arr.shape[-1] != len(names):
        raise ValueError(
            f"shape mismatch: values has {arr.shape[-1]} features, names has {len(names)}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "SHAP values contain NaN or Inf; check the explainer output"
        )

    return arr, names
