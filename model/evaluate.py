"""Evaluation metrics calculator for window and event-level evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics at sample/window level.

    Args:
        y_true: Array of true ground-truth binary labels (0 or 1).
        y_prob: Array of predicted probability scores in range [0, 1].
        threshold: Classification decision threshold.

    Returns:
        Dictionary of metrics: roc_auc, pr_auc, precision, recall, f1, accuracy.
    """
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    if len(y_true) == 0:
        return {"roc_auc": 0.0, "pr_auc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    y_pred = (y_prob >= threshold).astype(int)

    has_both_classes = len(np.unique(y_true)) > 1
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else 0.0

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "threshold": threshold,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.80,
) -> float:
    """Find decision threshold maximizing F1 while maintaining target recall.

    Args:
        y_true: True binary labels.
        y_prob: Probability predictions.
        target_recall: Desired minimum recall constraint.

    Returns:
        Optimal float threshold in range [0.05, 0.95].
    """
    best_thresh = 0.5
    best_f1 = -1.0

    for thresh in np.linspace(0.05, 0.95, 91):
        m = evaluate_predictions(y_true, y_prob, threshold=thresh)
        if m["recall"] >= target_recall and m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = thresh

    return float(best_thresh)


def compute_event_recall(
    metadata_list: list[dict[str, Any]],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate clip/event-level recall (if clip has >= 1 positive window, it is flagged).

    Args:
        metadata_list: List of metadata dicts containing 'clip_id'.
        y_true: Array of true labels per window.
        y_prob: Predicted probability scores.
        threshold: Alert threshold.

    Returns:
        Dictionary of event-level recall metrics.
    """
    if len(metadata_list) != len(y_true) or len(y_true) == 0:
        return {"event_recall": 0.0, "total_events": 0, "detected_events": 0}

    # Group windows by clip_id
    clips: dict[str, dict[str, Any]] = {}
    for meta, true_lbl, prob in zip(metadata_list, y_true, y_prob):
        c_id = meta.get("clip_id", "unknown")
        if c_id not in clips:
            clips[c_id] = {"label": true_lbl, "max_prob": prob}
        else:
            clips[c_id]["max_prob"] = max(clips[c_id]["max_prob"], prob)

    total_shoplifting_clips = sum(1 for c in clips.values() if c["label"] == 1)
    detected_shoplifting_clips = sum(1 for c in clips.values() if c["label"] == 1 and c["max_prob"] >= threshold)

    event_recall = (
        float(detected_shoplifting_clips / total_shoplifting_clips)
        if total_shoplifting_clips > 0
        else 0.0
    )

    return {
        "event_recall": event_recall,
        "total_events": total_shoplifting_clips,
        "detected_events": detected_shoplifting_clips,
    }
