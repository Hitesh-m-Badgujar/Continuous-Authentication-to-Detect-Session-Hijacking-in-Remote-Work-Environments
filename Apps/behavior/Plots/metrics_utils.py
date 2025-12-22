"""Common metrics utilities for behavioural trust plots.

This module does NOT assume any specific CSV layout. It works on raw
arrays (y_true, scores), where:
    - y_true: 1 = genuine, 0 = impostor
    - scores: higher = more genuine (e.g. trust in [0, 1])
"""

from __future__ import annotations

from typing import Sequence, Tuple, Dict

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def compute_far_frr(
    y_true: Sequence[int],
    scores: Sequence[float],
    thresholds: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FAR (False Accept Rate) and FRR (False Reject Rate) vs threshold.

    Assumes:
      - label 1 = genuine
      - label 0 = impostor
      - Accept if score >= threshold
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)

    genuine_mask = y_true == 1
    impostor_mask = y_true == 0

    if genuine_mask.sum() == 0 or impostor_mask.sum() == 0:
        raise ValueError("Need both genuine (1) and impostor (0) samples.")

    fars = []
    frrs = []

    for t in thresholds:
        accepted = scores >= t

        # FAR: impostors incorrectly accepted
        fa = np.sum(accepted & impostor_mask) / impostor_mask.sum()

        # FRR: genuine users incorrectly rejected
        fr = np.sum(~accepted & genuine_mask) / genuine_mask.sum()

        fars.append(fa)
        frrs.append(fr)

    return thresholds, np.asarray(fars), np.asarray(frrs)


def compute_eer(
    fars: np.ndarray,
    frrs: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and corresponding threshold."""
    fars = np.asarray(fars, dtype=float)
    frrs = np.asarray(frrs, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    diffs = np.abs(fars - frrs)
    idx = int(np.argmin(diffs))
    eer = float((fars[idx] + frrs[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold


def compute_confusion_at_threshold(
    y_true: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> Tuple[int, int, int, int]:
    """Compute TP, FP, TN, FN at a given threshold.

    Prediction rule:
      predict genuine if score >= threshold, impostor otherwise.
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    preds = (scores >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))

    return tp, fp, tn, fn


def compute_basic_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1 given a confusion matrix."""
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_roc(
    y_true: Sequence[int],
    scores: Sequence[float],
):
    """Compute ROC curve (FPR, TPR, thresholds) and AUC."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    fpr, tpr, thresh = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresh, float(roc_auc)


def compute_precision_recall(
    y_true: Sequence[int],
    scores: Sequence[float],
):
    """Compute Precision-Recall curve and approximate area under PR curve."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    precision, recall, thresh = precision_recall_curve(y_true, scores)

    # Approximate area under PR curve using trapezoidal rule.
    pr_area = float(np.trapz(precision[::-1], recall[::-1]))

    return precision, recall, thresh, pr_area
