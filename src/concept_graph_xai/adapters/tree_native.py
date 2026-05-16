"""Adapter from a model's native ``feature_importances_`` attribute."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def from_feature_importances_(
    model: Any,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Pull ``model.feature_importances_`` into the canonical ``(values, names)``."""

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{type(model).__name__} has no feature_importances_")
    values = np.asarray(model.feature_importances_, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"expected 1D array, got shape {values.shape}")
    names = list(feature_names)
    if values.shape[0] != len(names):
        raise ValueError(
            f"shape mismatch: values has {values.shape[0]} entries, names has {len(names)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "feature_importances_ contains NaN or Inf"
        )
    return values, names
