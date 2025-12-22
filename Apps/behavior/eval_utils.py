# Apps/behavior/eval_utils.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    auc,
)


@dataclass
class BinaryMetrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    far: float
    frr: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass
class RocPrAuc:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    precision_curve: np.ndarray
    recall_curve: np.ndarray
    pr_thresholds: np.ndarray
    roc_auc: float
    pr_auc: float
    eer: float
    eer_threshold: float


def compute_confusion_and_basic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> BinaryMetrics:
    """
    y_true: 1 = genuine, 0 = impostor
    y_pred: 1 = predicted genuine, 0 = predicted impostor
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    # use zero_division=0 to avoid crashes if a class is missing
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # FAR = FP / (FP + TN)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # FRR = FN / (FN + TP)
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # threshold will be filled in later / overridden
    return BinaryMetrics(
        threshold=np.nan,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        far=far,
        frr=frr,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def compute_roc_pr_eer(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> RocPrAuc:
    """
    scores: higher = more likely genuine
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)  # pos_label=1 by default

    # FRR = 1 - TPR
    fnr = 1.0 - tpr
    # find point where |FNR - FPR| is minimal
    abs_diff = np.abs(fnr - fpr)
    idx = np.argmin(abs_diff)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    eer_threshold = thresholds[idx]

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, scores
    )
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall_curve, precision_curve)

    return RocPrAuc(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        precision_curve=precision_curve,
        recall_curve=recall_curve,
        pr_thresholds=pr_thresholds,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        eer=eer,
        eer_threshold=eer_threshold,
    )


def threshold_scores(
    scores: np.ndarray, threshold: float
) -> np.ndarray:
    """
    scores: higher = genuine
    """
    return (scores >= threshold).astype(int)


def find_best_threshold_for_f1(
    y_true: np.ndarray,
    scores: np.ndarray,
    num_candidates: int = 200,
) -> Tuple[float, BinaryMetrics]:
    """
    Simple grid search over score range to maximise F1.
    """
    min_s, max_s = scores.min(), scores.max()
    if min_s == max_s:
        # degenerate; any threshold gives the same predictions
        thr = 0.5 * (min_s + max_s)
        y_pred = threshold_scores(scores, thr)
        metrics = compute_confusion_and_basic_metrics(y_true, y_pred)
        metrics.threshold = thr
        return thr, metrics

    thresholds = np.linspace(min_s, max_s, num_candidates)
    best_f1 = -1.0
    best_thr = thresholds[0]
    best_metrics = None

    for thr in thresholds:
        y_pred = threshold_scores(scores, thr)
        metrics = compute_confusion_and_basic_metrics(y_true, y_pred)
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_thr = thr
            best_metrics = metrics

    best_metrics.threshold = best_thr
    return best_thr, best_metrics
