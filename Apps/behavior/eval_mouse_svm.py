# Apps/behavior/eval_mouse_svm.py

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eval_utils import (
    compute_roc_pr_eer,
    find_best_threshold_for_f1,
    threshold_scores,
    compute_confusion_and_basic_metrics,
)

DATA_CSV = "Data\mouse_windows_test.csv"
SCALER_PATH = "artifacts/mouse_svm_scaler.pkl"
MODEL_PATH = "artifacts/mouse_svm.pkl"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def eval_mouse():
    df = pd.read_csv(DATA_CSV)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        svm = pickle.load(f)

    X_scaled = scaler.transform(X)
    scores = svm.decision_function(X_scaled)

    roc_info = compute_roc_pr_eer(y, scores)
    thr_f1, metrics_at_f1 = find_best_threshold_for_f1(y, scores)

    # Also compute metrics at EER threshold
    y_pred_eer = threshold_scores(scores, roc_info.eer_threshold)
    metrics_at_eer = compute_confusion_and_basic_metrics(y, y_pred_eer)
    metrics_at_eer.threshold = roc_info.eer_threshold

    # Save global mouse metrics (Table 4.4)
    rows = [
        {
            "operating_point": "best_F1",
            "threshold": metrics_at_f1.threshold,
            "accuracy": metrics_at_f1.accuracy,
            "precision": metrics_at_f1.precision,
            "recall": metrics_at_f1.recall,
            "f1": metrics_at_f1.f1,
            "far": metrics_at_f1.far,
            "frr": metrics_at_f1.frr,
            "eer": roc_info.eer,
            "eer_threshold": roc_info.eer_threshold,
            "roc_auc": roc_info.roc_auc,
            "pr_auc": roc_info.pr_auc,
            "tp": metrics_at_f1.tp,
            "fp": metrics_at_f1.fp,
            "tn": metrics_at_f1.tn,
            "fn": metrics_at_f1.fn,
        },
        {
            "operating_point": "eer_threshold",
            "threshold": metrics_at_eer.threshold,
            "accuracy": metrics_at_eer.accuracy,
            "precision": metrics_at_eer.precision,
            "recall": metrics_at_eer.recall,
            "f1": metrics_at_eer.f1,
            "far": metrics_at_eer.far,
            "frr": metrics_at_eer.frr,
            "eer": roc_info.eer,
            "eer_threshold": roc_info.eer_threshold,
            "roc_auc": roc_info.roc_auc,
            "pr_auc": roc_info.pr_auc,
            "tp": metrics_at_eer.tp,
            "fp": metrics_at_eer.fp,
            "tn": metrics_at_eer.tn,
            "fn": metrics_at_eer.fn,
        },
    ]

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(RESULTS_DIR, "mouse_svm_metrics.csv"), index=False)

    # ROC + PR plots – Figure 4.3
    plt.figure()
    plt.plot(roc_info.fpr, roc_info.tpr, label=f"Mouse SVM (AUC={roc_info.roc_auc:.3f})")
    plt.scatter(
        roc_info.eer,
        1.0 - roc_info.eer,
        marker="x",
        label=f"EER={roc_info.eer:.3f}",
    )
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Mouse SVM ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_mouse_svm_roc.png"))
    plt.close()

    plt.figure()
    plt.plot(
        roc_info.recall_curve,
        roc_info.precision_curve,
        label=f"Mouse SVM (AUC={roc_info.pr_auc:.3f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Mouse SVM PR Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_mouse_svm_pr.png"))
    plt.close()

    print("Mouse SVM evaluation saved to:", RESULTS_DIR)


if __name__ == "__main__":
    eval_mouse()
