"""Adapter from ``sklearn.inspection.permutation_importance`` results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def from_permutation_importance(
    result: Any,
    feature_names: Sequence[str],
    *,
    use: str = "importances_mean",
) -> tuple[np.ndarray, list[str]]:
    """Convert a sklearn ``Bunch`` (from ``permutation_importance``) to arrays.

    Parameters
    ----------
    result:
        The Bunch returned by ``sklearn.inspection.permutation_importance``,
        with attributes ``importances_mean`` and ``importances_std``.
    feature_names:
        Names matching the order of features used during the permutation run.
    use:
        Which attribute on the Bunch to expose. Defaults to ``importances_mean``.
    """

    if not hasattr(result, use):
        raise AttributeError(f"permutation_importance result has no attribute {use!r}")
    values = np.asarray(getattr(result, use), dtype=float)
    if values.ndim != 1:
        raise ValueError(f"expected 1D array, got shape {values.shape}")
    names = list(feature_names)
    if values.shape[0] != len(names):
        raise ValueError(
            f"shape mismatch: values has {values.shape[0]} entries, names has {len(names)}"
        )
    return values, names
