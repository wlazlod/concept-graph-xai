"""Adapters that convert various importance sources to canonical arrays.

Every adapter returns a tuple ``(values, feature_names)`` where:

* ``values`` is a ``numpy.ndarray`` shaped ``(F,)`` (per-feature aggregate) or
  ``(N, F)`` (per-sample) of float dtype;
* ``feature_names`` is a ``list[str]`` of length ``F``.
"""

from concept_graph_xai.adapters.shap import from_shap_explanation
from concept_graph_xai.adapters.sklearn_perm import from_permutation_importance
from concept_graph_xai.adapters.tree_native import from_feature_importances_

__all__ = [
    "from_feature_importances_",
    "from_permutation_importance",
    "from_shap_explanation",
]
