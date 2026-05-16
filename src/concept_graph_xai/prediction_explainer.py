"""Single-prediction explainer rolled up to concept level (P5).

Given a fitted model and per-sample SHAP values, this class explains *one*
prediction using the supplied concept tree. The headline plot is
:meth:`ConceptPredictionExplainer.waterfall` — like ``shap.plots.waterfall``
but with each concept's contribution summed up the tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from concept_graph_xai.graph import ConceptGraph


@dataclass(frozen=True)
class ConceptContribution:
    """One row of the per-concept breakdown for a single prediction."""

    name: str
    path: str
    depth: int
    feature_count: int
    shap_sum: float


class ConceptPredictionExplainer:
    """Explain a single prediction using the concept tree.

    Parameters
    ----------
    graph:
        The ConceptGraph.
    model:
        A fitted model. Used only to display the prediction alongside the
        waterfall; the SHAP values are the source of truth.
    X:
        The feature matrix the SHAP values were computed on. Must be a
        DataFrame whose columns include every feature in the graph.
    shap_values:
        Per-sample SHAP values of shape ``(N, F)`` aligned with ``X``'s
        columns.
    base_value:
        The SHAP base value (typically ``shap.TreeExplainer(model).expected_value``).
        Required so the waterfall can show the absolute level.
    feature_names:
        Optional override for ``X.columns``. Useful when the SHAP values were
        produced from a numpy array.
    """

    def __init__(
        self,
        graph: ConceptGraph,
        model: Any,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        base_value: float,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        arr = np.asarray(shap_values, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"shap_values must be 2D (N, F); got {arr.shape}")
        if arr.shape[0] != len(X):
            raise ValueError(
                f"shap_values has {arr.shape[0]} rows but X has {len(X)}"
            )

        names = list(feature_names) if feature_names is not None else list(X.columns)
        if arr.shape[1] != len(names):
            raise ValueError(
                f"shap_values has {arr.shape[1]} columns but feature_names has {len(names)}"
            )

        self.graph = graph
        self.model = model
        self.X = X
        self.shap_values = arr
        self.feature_names = names
        self.base_value = float(base_value)
        self._name_to_idx = {n: i for i, n in enumerate(names)}

    # ------------------------------------------------------------------ #
    # Per-row breakdown
    # ------------------------------------------------------------------ #
    def _resolve_row(self, row: int | str) -> int:
        if isinstance(row, int):
            if row < 0 or row >= len(self.X):
                raise IndexError(f"row index {row} out of range [0, {len(self.X)})")
            return row
        try:
            pos = self.X.index.get_loc(row)
        except KeyError as exc:
            raise KeyError(f"row label {row!r} not found in X.index") from exc
        if isinstance(pos, slice | np.ndarray):
            raise KeyError(f"row label {row!r} is not unique in X.index")
        return int(pos)

    def _concepts_at_depth(self, depth: int) -> list[str]:
        """Return concepts whose distance from the root equals ``depth``."""

        if depth < 1:
            raise ValueError("depth must be >= 1 (root-only is uninformative)")
        out = [
            n
            for n in self.graph.nodes_in_order()
            if self.graph.kind(n) == "concept" and len(self.graph.path(n)) - 1 == depth
        ]
        return out

    def breakdown(self, row: int | str, *, depth: int = 1) -> pd.DataFrame:
        """Per-concept SHAP sum for a single prediction at the given depth.

        Parameters
        ----------
        row:
            Either a positional index (``int``) or a label in ``X.index``.
        depth:
            Depth of the concepts to roll up to. ``depth=1`` aggregates each
            top-level concept (children of the root); higher depths reveal
            more granular concepts.

        Returns
        -------
        pandas.DataFrame
            Sorted by ``shap_sum`` descending. Columns: ``name``, ``path``,
            ``depth``, ``feature_count``, ``shap_sum``.
        """

        idx = self._resolve_row(row)
        shap_row = self.shap_values[idx]
        concepts = self._concepts_at_depth(depth)
        if not concepts:
            raise ValueError(f"no concepts at depth {depth}")

        rows: list[ConceptContribution] = []
        for node in concepts:
            feats = [f for f in self.graph.descendant_features(node) if f in self._name_to_idx]
            if not feats:
                continue
            idxs = [self._name_to_idx[f] for f in feats]
            shap_sum = float(shap_row[idxs].sum())
            rows.append(
                ConceptContribution(
                    name=node,
                    path="/".join(self.graph.path(node)),
                    depth=depth,
                    feature_count=len(feats),
                    shap_sum=shap_sum,
                )
            )

        df = pd.DataFrame([r.__dict__ for r in rows])
        if df.empty:
            return df
        return df.sort_values("shap_sum", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    def waterfall(
        self,
        row: int | str,
        *,
        depth: int = 1,
        title: str | None = None,
        layout_kwargs: dict[str, Any] | None = None,
    ) -> go.Figure:
        """Render a per-prediction concept waterfall.

        The chart starts at ``base_value``, applies each concept's SHAP sum in
        descending magnitude, and ends at the predicted logit
        (``base_value + sum(SHAP)``). Bars are coloured red for negative
        contributions and green for positive.

        Parameters
        ----------
        row:
            Row index or label.
        depth:
            Tree depth to roll up to. See :meth:`breakdown`.
        title:
            Figure title.
        layout_kwargs:
            Passed verbatim to ``fig.update_layout``.

        Returns
        -------
        plotly.graph_objects.Figure
        """

        idx = self._resolve_row(row)
        # breakdown() raises ValueError("no concepts at depth ...") on the
        # only failure mode that produces an empty frame, so no defensive
        # empty-check is needed here.
        df = self.breakdown(row, depth=depth)

        labels = ["base", *df["name"].tolist(), "prediction"]
        values = [self.base_value, *df["shap_sum"].tolist(), 0.0]
        measures = ["absolute", *(["relative"] * len(df)), "total"]

        prediction_logit = float(self.base_value + df["shap_sum"].sum())
        proba = float("nan")
        if self.model is not None and hasattr(self.model, "predict_proba"):
            proba_arr = np.asarray(self.model.predict_proba(self.X.iloc[[idx]]))
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                proba = float(proba_arr[0, 1])

        fig = go.Figure(
            go.Waterfall(
                orientation="h",
                y=labels,
                x=values,
                measure=measures,
                connector={"line": {"color": "rgba(80,80,80,0.4)"}},
                decreasing={"marker": {"color": "#d62728"}},
                increasing={"marker": {"color": "#2ca02c"}},
                totals={"marker": {"color": "#1f77b4"}},
                textposition="outside",
                texttemplate="%{x:+.3f}",
            )
        )

        subtitle = (
            f"row={row}  ·  base={self.base_value:+.3f}  "
            f"·  predicted logit={prediction_logit:+.3f}"
        )
        if not np.isnan(proba):
            subtitle += f"  ·  P(y=1)={proba:.3f}"

        fig.update_layout(
            title=title or f"Concept waterfall — {subtitle}",
            xaxis_title="SHAP contribution (logit space)",
            yaxis={"autorange": "reversed"},
            margin={"t": 60, "l": 160, "r": 60, "b": 60},
            height=max(300, 35 * (len(df) + 2) + 100),
        )
        if layout_kwargs:
            fig.update_layout(**layout_kwargs)
        return fig
