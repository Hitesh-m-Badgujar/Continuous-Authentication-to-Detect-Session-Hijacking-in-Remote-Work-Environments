# Apps/behavior/eval_fusion_behavioral.py

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

RESULTS_DIR = "results"
FUSION_DATA = "Data/fusion_test_windows.csv"
KB_SVM_DIR = "artifacts/keyboard_svm"
MOUSE_SCALER_PATH = "artifacts/mouse_svm_scaler.pkl"
MOUSE_MODEL_PATH = "artifacts/mouse_svm.pkl"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Weights chosen based on validation – adapt if you have actual values
W_KEYBOARD = 0.7
W_MOUSE = 0.3


def eval_fusion():
    df = pd.read_csv(FUSION_DATA)

    y = df["label"].values.astype(int)
    users = df["user_id"].astype(str).values

    # Separate feature spaces
    kb_feature_cols = [c for c in df.columns if c.startswith("kb_")]
    mouse_feature_cols = [c for c in df.columns if c.startswith("mouse_")]

    X_kb = df[kb_feature_cols].values.astype(float)
    X_mouse = df[mouse_feature_cols].values.astype(float)

    # -------- Keyboard SVM per-user ----------
    kb_scores = np.zeros_like(y, dtype=float)

    unique_users = sorted(np.unique(users))
    for uid in unique_users:
        mask = (users == uid)
        X_u = X_kb[mask]

        scaler_path = os.path.join(KB_SVM_DIR, f"{uid}_scaler.pkl")
        model_path = os.path.join(KB_SVM_DIR, f"{uid}_svm.pkl")

        if not (os.path.exists(scaler_path) and os.path.exists(model_path)):
            print(f"[WARN] Missing keyboard SVM for user {uid}, skipping.")
            continue

        with open(scaler_path, "rb") as f:
            kb_scaler = pickle.load(f)
        with open(model_path, "rb") as f:
            kb_svm = pickle.load(f)

        X_u_scaled = kb_scaler.transform(X_u)
        kb_scores[mask] = kb_svm.decision_function(X_u_scaled)

    # -------- Mouse SVM global ----------
    with open(MOUSE_SCALER_PATH, "rb") as f:
        mouse_scaler = pickle.load(f)
    with open(MOUSE_MODEL_PATH, "rb") as f:
        mouse_svm = pickle.load(f)

    X_mouse_scaled = mouse_scaler.transform(X_mouse)
    mouse_scores = mouse_svm.decision_function(X_mouse_scaled)

    # -------- Fused score ----------
    fused_scores = W_KEYBOARD * kb_scores + W_MOUSE * mouse_scores

    # Evaluate each (keyboard-only, mouse-only, fusion)
    rows = []
    curves = {}

    for name, scores in [
        ("keyboard_only", kb_scores),
        ("mouse_only", mouse_scores),
        ("kb_mouse_fusion", fused_scores),
    ]:
        roc_info = compute_roc_pr_eer(y, scores)
        thr_f1, metrics_at_f1 = find_best_threshold_for_f1(y, scores)

        rows.append({
            "model": name,
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
        })

        curves[name] = roc_info

    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(RESULTS_DIR, "fusion_keyboard_mouse_metrics.csv"), index=False)

    # -------- Plots: Figure 4.5 – ROC & PR curves ----------
    plt.figure()
    for name, roc_info in curves.items():
        plt.plot(roc_info.fpr, roc_info.tpr, label=f"{name} (AUC={roc_info.roc_auc:.3f})")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Keyboard vs Mouse vs Fusion – ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_fusion_keyboard_mouse_roc.png"))
    plt.close()

    plt.figure()
    for name, roc_info in curves.items():
        plt.plot(
            roc_info.recall_curve,
            roc_info.precision_curve,
            label=f"{name} (AUC={roc_info.pr_auc:.3f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Keyboard vs Mouse vs Fusion – PR")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_fusion_keyboard_mouse_pr.png"))
    plt.close()

    print("Fusion evaluation saved to:", RESULTS_DIR)


if __name__ == "__main__":
    eval_fusion()
