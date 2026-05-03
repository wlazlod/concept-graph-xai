"""Concept-level AUC-loss metric (idea #3 from the scoping doc).

Three strategies:

* ``permutation`` — shuffle the values of each concept's features across rows
  (n_repeats), score on the held-out set, drop = baseline - mean(score). Cheap,
  model-agnostic, no retraining.
* ``retrain`` — drop those columns from the training set, retrain via a
  user-supplied ``train_fn(X_train, y_train) -> fitted_estimator``, and score on
  the held-out set with the same columns dropped. Most faithful to "lack of data".
* ``shap_marginal`` — subtract the concept's SHAP contributions from logits,
  re-apply the link function, score. Cheapest; an approximation under SHAP
  additivity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer, roc_auc_score

from concept_graph_xai.graph import ConceptGraph
from concept_graph_xai.metrics._common import empty_concept_frame

Strategy = Literal["permutation", "retrain", "shap_marginal"]
ScorerLike = str | Callable[[np.ndarray, np.ndarray], float]


def _resolve_scorer(metric: ScorerLike) -> Callable[[np.ndarray, np.ndarray], float]:
    if callable(metric):
        return metric
    if metric == "roc_auc":
        return lambda y, p: float(roc_auc_score(y, p))
    scorer = get_scorer(metric)

    def _call(y: np.ndarray, p: np.ndarray) -> float:
        score_func = scorer._score_func
        kwargs = scorer._kwargs
        return float(score_func(y, p, **kwargs))

    return _call


def _proba(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba_arr = np.asarray(proba, dtype=float)
        if proba_arr.ndim == 2 and proba_arr.shape[1] == 2:
            return proba_arr[:, 1]
        if proba_arr.ndim == 1:
            return proba_arr
        raise ValueError(
            f"binary classification expected; predict_proba returned shape {proba_arr.shape}"
        )
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    raise AttributeError(f"{type(model).__name__} has no predict_proba or decision_function")


def _features_in_x(
    X: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str],
) -> list[str]:
    if isinstance(X, pd.DataFrame):
        return list(X.columns)
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length {len(feature_names)} != X.shape[1] {X.shape[1]}"
        )
    return list(feature_names)


def _structural_feature_counts(
    graph: ConceptGraph,
    feats_in_X_set: set[str],
) -> np.ndarray:
    """Per-node count of descendant features intersected with ``feats_in_X_set``.

    Always populated for every node (including the root), regardless of any
    ``skip_root`` flag — sector size in the resulting sunburst depends on it,
    and Plotly's ``branchvalues='total'`` silently drops the chart when a
    parent's value is smaller than the sum of its children's values.
    """

    counts = np.zeros(len(graph), dtype=int)
    for i, node in enumerate(graph.nodes_in_order()):
        feats = [f for f in graph.descendant_features(node) if f in feats_in_X_set]
        counts[i] = len(feats)
    return counts


def _permute_features(
    X: pd.DataFrame | np.ndarray,
    cols: Sequence[str],
    feature_names: Sequence[str],
    rng: np.random.Generator,
) -> pd.DataFrame | np.ndarray:
    if isinstance(X, pd.DataFrame):
        permuted = X.copy()
        for col in cols:
            permuted[col] = rng.permutation(permuted[col].to_numpy())
        return permuted
    arr = np.asarray(X, dtype=float).copy()
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for col in cols:
        idx = name_to_idx[col]
        arr[:, idx] = rng.permutation(arr[:, idx])
    return arr


def _drop_features(
    X: pd.DataFrame | np.ndarray,
    cols: Sequence[str],
    feature_names: Sequence[str],
) -> tuple[pd.DataFrame | np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        kept = [c for c in X.columns if c not in set(cols)]
        return X.loc[:, kept], kept
    drop_idx = {feature_names.index(c) for c in cols}
    keep_idx = [i for i in range(len(feature_names)) if i not in drop_idx]
    arr = np.asarray(X, dtype=float)[:, keep_idx]
    return arr, [feature_names[i] for i in keep_idx]


def auc_drop(
    graph: ConceptGraph,
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray | pd.Series,
    feature_names: Sequence[str] | None = None,
    *,
    strategy: Strategy = "permutation",
    metric: ScorerLike = "roc_auc",
    n_repeats: int = 10,
    random_state: int | None = 42,
    train_fn: Callable[[pd.DataFrame | np.ndarray, np.ndarray], Any] | None = None,
    X_train: pd.DataFrame | np.ndarray | None = None,
    y_train: np.ndarray | pd.Series | None = None,
    shap_values: np.ndarray | None = None,
    base_predictions: np.ndarray | None = None,
    skip_root: bool = True,
) -> pd.DataFrame:
    """Compute concept-level metric drop under ablation."""

    feats_in_X = _features_in_x(X, feature_names or list(graph.features()))
    y_arr = np.asarray(y)
    score = _resolve_scorer(metric)

    if strategy == "shap_marginal":
        df = _shap_marginal(graph, X, y_arr, score, shap_values, base_predictions, skip_root)
    elif strategy == "permutation":
        df = _permutation(
            graph, model, X, y_arr, feats_in_X, score, n_repeats, random_state, skip_root
        )
    elif strategy == "retrain":
        df = _retrain(
            graph,
            X,
            y_arr,
            feats_in_X,
            score,
            train_fn,
            X_train,
            y_train,
            skip_root,
        )
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    df["strategy"] = strategy
    return df


# ---------------------------------------------------------------------- #
# Permutation
# ---------------------------------------------------------------------- #
def _permutation(
    graph: ConceptGraph,
    model: Any,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    feats_in_X: Sequence[str],
    score: Callable[[np.ndarray, np.ndarray], float],
    n_repeats: int,
    random_state: int | None,
    skip_root: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    base_pred = _proba(model, X)
    baseline = float(score(y, base_pred))

    df = empty_concept_frame(graph)
    n_nodes = len(df)
    drop_mean = np.full(n_nodes, np.nan, dtype=float)
    drop_std = np.full(n_nodes, np.nan, dtype=float)
    ablated_mean = np.full(n_nodes, np.nan, dtype=float)

    feats_in_X_set = set(feats_in_X)
    feature_count = _structural_feature_counts(graph, feats_in_X_set)
    for i, node in enumerate(graph.nodes_in_order()):
        if skip_root and node == graph.root:
            continue
        feats = [f for f in graph.descendant_features(node) if f in feats_in_X_set]
        if not feats:
            continue
        scores: list[float] = []
        for _ in range(n_repeats):
            X_perm = _permute_features(X, feats, feats_in_X, rng)
            preds = _proba(model, X_perm)
            scores.append(float(score(y, preds)))
        scores_arr = np.asarray(scores, dtype=float)
        ablated_mean[i] = float(scores_arr.mean())
        drop_mean[i] = baseline - ablated_mean[i]
        drop_std[i] = float(scores_arr.std(ddof=0))

    df["feature_count"] = feature_count
    df["baseline_score"] = baseline
    df["ablated_score_mean"] = ablated_mean
    df["auc_drop_mean"] = drop_mean
    df["auc_drop_std"] = drop_std
    return df


# ---------------------------------------------------------------------- #
# Retrain
# ---------------------------------------------------------------------- #
def _retrain(
    graph: ConceptGraph,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    feats_in_X: Sequence[str],
    score: Callable[[np.ndarray, np.ndarray], float],
    train_fn: Callable[[pd.DataFrame | np.ndarray, np.ndarray], Any] | None,
    X_train: pd.DataFrame | np.ndarray | None,
    y_train: np.ndarray | pd.Series | None,
    skip_root: bool,
) -> pd.DataFrame:
    if train_fn is None:
        raise ValueError("strategy='retrain' requires train_fn=...")
    if X_train is None or y_train is None:
        raise ValueError("strategy='retrain' requires X_train and y_train")
    y_train_arr = np.asarray(y_train)

    base_model = train_fn(X_train, y_train_arr)
    base_pred = _proba(base_model, X)
    baseline = float(score(y, base_pred))

    df = empty_concept_frame(graph)
    n_nodes = len(df)
    drop_mean = np.full(n_nodes, np.nan, dtype=float)
    ablated_mean = np.full(n_nodes, np.nan, dtype=float)

    feats_in_X_set = set(feats_in_X)
    feature_count = _structural_feature_counts(graph, feats_in_X_set)
    for i, node in enumerate(graph.nodes_in_order()):
        if skip_root and node == graph.root:
            continue
        feats = [f for f in graph.descendant_features(node) if f in feats_in_X_set]
        if not feats:
            continue
        X_train_drop, _kept = _drop_features(X_train, feats, feats_in_X)
        X_test_drop, _kept2 = _drop_features(X, feats, feats_in_X)
        ablated = train_fn(X_train_drop, y_train_arr)
        preds = _proba(ablated, X_test_drop)
        ablated_mean[i] = float(score(y, preds))
        drop_mean[i] = baseline - ablated_mean[i]

    df["feature_count"] = feature_count
    df["baseline_score"] = baseline
    df["ablated_score_mean"] = ablated_mean
    df["auc_drop_mean"] = drop_mean
    df["auc_drop_std"] = np.nan
    return df


# ---------------------------------------------------------------------- #
# SHAP-marginal
# ---------------------------------------------------------------------- #
def _logit(p: np.ndarray) -> np.ndarray:
    eps = 1e-9
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=float)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=float)


def _shap_marginal(
    graph: ConceptGraph,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray], float],
    shap_values: np.ndarray | None,
    base_predictions: np.ndarray | None,
    skip_root: bool,
) -> pd.DataFrame:
    if shap_values is None or base_predictions is None:
        raise ValueError(
            "strategy='shap_marginal' requires shap_values=(N,F) and base_predictions=(N,)"
        )
    shap_arr = np.asarray(shap_values, dtype=float)
    if shap_arr.ndim != 2:
        raise ValueError(f"shap_values must be 2D (N,F), got {shap_arr.shape}")

    feats_in_X = _features_in_x(X, list(graph.features()))
    if shap_arr.shape[1] != len(feats_in_X):
        raise ValueError(
            f"shap_values has {shap_arr.shape[1]} columns; feature names imply {len(feats_in_X)}"
        )

    base_pred = np.asarray(base_predictions, dtype=float)
    base_logit = _logit(base_pred)
    baseline = float(score(y, base_pred))

    name_to_idx = {n: i for i, n in enumerate(feats_in_X)}
    feats_in_X_set = set(feats_in_X)

    df = empty_concept_frame(graph)
    n_nodes = len(df)
    drop_mean = np.full(n_nodes, np.nan, dtype=float)
    ablated_mean = np.full(n_nodes, np.nan, dtype=float)
    feature_count = _structural_feature_counts(graph, feats_in_X_set)

    for i, node in enumerate(graph.nodes_in_order()):
        if skip_root and node == graph.root:
            continue
        feats = [f for f in graph.descendant_features(node) if f in feats_in_X_set]
        if not feats:
            continue
        idxs = [name_to_idx[f] for f in feats]
        contribution = shap_arr[:, idxs].sum(axis=1)
        new_logit = base_logit - contribution
        new_pred = _sigmoid(new_logit)
        ablated_mean[i] = float(score(y, new_pred))
        drop_mean[i] = baseline - ablated_mean[i]

    df["feature_count"] = feature_count
    df["baseline_score"] = baseline
    df["ablated_score_mean"] = ablated_mean
    df["auc_drop_mean"] = drop_mean
    df["auc_drop_std"] = np.nan
    return df
