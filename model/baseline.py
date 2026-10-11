"""Baseline tabular classifier (Logistic Regression / Random Forest) on aggregated sequence features."""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def aggregate_sequence_features(X_seq: np.ndarray) -> np.ndarray:
    """Aggregate (N, T, F) sequence tensors into (N, 5*F) 2D tabular features.

    Computes mean, std, min, max, and last-frame values across time axis T.

    Args:
        X_seq: Sequence array of shape (N, T, F).

    Returns:
        Aggregated tabular feature array of shape (N, 5 * F).
    """
    if X_seq.ndim != 3 or len(X_seq) == 0:
        return np.empty((0, 0), dtype=np.float32)

    mean_f = np.mean(X_seq, axis=1)
    std_f = np.std(X_seq, axis=1)
    min_f = np.min(X_seq, axis=1)
    max_f = np.max(X_seq, axis=1)
    last_f = X_seq[:, -1, :]

    return np.hstack([mean_f, std_f, min_f, max_f, last_f])


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "logistic",
    seed: int = 42,
) -> Pipeline:
    """Train a tabular baseline pipeline (Scaler + Classifier).

    Args:
        X_train: 3D sequence array (N, T, F) or 2D aggregated matrix.
        y_train: Label array (N,).
        model_type: "logistic" or "random_forest".
        seed: Random seed.

    Returns:
        Fitted sklearn Pipeline.
    """
    if X_train.ndim == 3:
        X_agg = aggregate_sequence_features(X_train)
    else:
        X_agg = X_train

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed)
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])

    pipeline.fit(X_agg, y_train)
    return pipeline


def evaluate_baseline_model(
    pipeline: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate a trained baseline pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_test: 3D sequence array (N, T, F) or 2D matrix.
        y_test: True labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    if len(X_test) == 0:
        return {"roc_auc": 0.0, "pr_auc": 0.0, "accuracy": 0.0}

    if X_test.ndim == 3:
        X_agg = aggregate_sequence_features(X_test)
    else:
        X_agg = X_test

    y_prob = pipeline.predict_proba(X_agg)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0
    pr_auc = float(average_precision_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0
    accuracy = float(np.mean(y_pred == y_test))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
    }


def save_baseline_model(pipeline: Pipeline, path: str | Path) -> None:
    """Save baseline model pipeline to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_baseline_model(path: str | Path) -> Pipeline:
    """Load baseline model pipeline from disk."""
    return joblib.load(path)
