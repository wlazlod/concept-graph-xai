"""Count-based metric: how many features each concept covers."""

from __future__ import annotations

import pandas as pd

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import empty_concept_frame


def feature_counts(graph: ConceptGraph) -> pd.DataFrame:
    """Return a tidy DataFrame with one row per node and a ``count`` column.

    For a feature node, ``count = 1``. For a concept node, ``count`` is the
    number of feature descendants of that concept.
    """

    df = empty_concept_frame(graph)
    counts = [len(graph.descendant_features(n)) for n in graph.nodes_in_order()]
    df["count"] = counts
    return df
