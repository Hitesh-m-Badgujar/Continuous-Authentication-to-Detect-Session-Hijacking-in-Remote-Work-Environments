# Apps/behavior/eval_keyboard_models.py

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eval_utils import (
    compute_roc_pr_eer,
    find_best_threshold_for_f1,
)

# ---------- CONFIG ----------
DATA_CSV = "Data/keyboard_test_windows.csv"
SVM_DIR = "artifacts/keyboard_svm"
AE_DIR = "artifacts/keyboard_ae"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# --------- AUTOENCODER HELPERS (STUBS) ---------
def load_ae_model(user_id: str):
    """
    Stub: adapt this to your AE framework (PyTorch, Keras, etc.) if you
    actually have AE models saved. For now this will not be used.
    """
    raise NotImplementedError("AE loading not implemented / not used.")


def ae_reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    """
    Stub for AE reconstruction error.
    """
    raise NotImplementedError("AE reconstruction not implemented / not used.")


# ---------- MAIN EVAL ----------

def eval_keyboard_models():
    # Load full test CSV
    df = pd.read_csv(DATA_CSV)

    # We NEED these columns to exist in the CSV:
    # user_id, label + numeric feature columns
    if "user_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Expected columns 'user_id' and 'label' in {DATA_CSV}, "
            f"but got: {list(df.columns)}"
        )

    # Extract user IDs and labels first (keep them as they are)
    users = df["user_id"].astype(str).values
    y_all = df["label"].values.astype(int)

    # Feature columns = all columns except IDs / label / split
    feature_cols = [
        c
        for c in df.columns
        if c not in ["user_id", "label", "session_id", "window_id", "split"]
    ]

    # Sanity check
    if not feature_cols:
        raise ValueError(
            "No feature columns found. Check that keyboard_test_windows.csv "
            "contains timing features like dwell_mean, dd_mean, etc."
        )

    X_all = df[feature_cols].values.astype(float)

    # Global containers for scores
    svm_scores = np.zeros_like(y_all, dtype=float)
    ae_scores = np.zeros_like(y_all, dtype=float)

    per_user_rows = []

    unique_users = sorted(np.unique(users))

    for user_id in unique_users:
        mask = (users == user_id)
        X_u = X_all[mask]
        y_u = y_all[mask]

        # ----- SVM -----
        svm_scaler_path = os.path.join(SVM_DIR, f"{user_id}_scaler.pkl")
        svm_model_path = os.path.join(SVM_DIR, f"{user_id}_svm.pkl")

        if not (os.path.exists(svm_scaler_path) and os.path.exists(svm_model_path)):
            print(f"[WARN] Missing SVM artifacts for user {user_id}, skipping this user.")
            continue

        with open(svm_scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(svm_model_path, "rb") as f:
            svm = pickle.load(f)

        # Scale and score
        X_u_scaled = scaler.transform(X_u)
        svm_score_u = svm.decision_function(X_u_scaled)
        svm_scores[mask] = svm_score_u

        # Per-user ROC/EER/F1 for SVM
        roc_info = compute_roc_pr_eer(y_u, svm_score_u)
        thr_f1, metrics_f1 = find_best_threshold_for_f1(y_u, svm_score_u)

        per_user_rows.append(
            {
                "user_id": user_id,
                "n_samples": int(len(y_u)),
                "accuracy": metrics_f1.accuracy,
                "precision": metrics_f1.precision,
                "recall": metrics_f1.recall,
                "f1": metrics_f1.f1,
                "far": metrics_f1.far,
                "frr": metrics_f1.frr,
                "eer": roc_info.eer,
                "eer_threshold": roc_info.eer_threshold,
                "best_f1_threshold": thr_f1,
                "tp": metrics_f1.tp,
                "fp": metrics_f1.fp,
                "tn": metrics_f1.tn,
                "fn": metrics_f1.fn,
            }
        )

        # ----- AE (optional) -----
        # If you don't have AE artifacts, this part will effectively do nothing.
        try:
            ae_scaler_path = os.path.join(AE_DIR, f"{user_id}_scaler.pkl")
            if os.path.exists(ae_scaler_path):
                with open(ae_scaler_path, "rb") as f:
                    ae_scaler = pickle.load(f)
                X_u_scaled_for_ae = ae_scaler.transform(X_u)
            else:
                X_u_scaled_for_ae = X_u  # assume pre-scaled

            ae_model_path = os.path.join(AE_DIR, f"{user_id}_ae.h5")
            if os.path.exists(ae_model_path):
                ae_model = load_ae_model(user_id)
                recon_err = ae_reconstruction_error(ae_model, X_u_scaled_for_ae)
                # Higher = more genuine (flip sign)
                ae_scores[mask] = -recon_err
            else:
                # No AE model for this user, skip
                pass
        except NotImplementedError:
            # AE not implemented; ignore
            pass

    # ---------- GLOBAL SVM METRICS ----------
    svm_roc = compute_roc_pr_eer(y_all, svm_scores)
    thr_f1_svm, svm_metrics_at_f1 = find_best_threshold_for_f1(y_all, svm_scores)

    # ---------- GLOBAL AE METRICS (optional) ----------
    ae_has_any = not np.allclose(ae_scores, 0.0)
    if ae_has_any:
        ae_roc = compute_roc_pr_eer(y_all, ae_scores)
        thr_f1_ae, ae_metrics_at_f1 = find_best_threshold_for_f1(y_all, ae_scores)
    else:
        ae_roc = None
        thr_f1_ae = None
        ae_metrics_at_f1 = None

    # ---------- SAVE TABLES ----------
    # Per-user table (Table 4.3)
    per_user_df = pd.DataFrame(per_user_rows)
    per_user_df.to_csv(
        os.path.join(RESULTS_DIR, "keyboard_svm_per_user.csv"), index=False
    )

    # Global AE vs SVM (Table 4.2)
    rows = [
        {
            "model": "keyboard_svm",
            "metric_operating_point": "best_F1",
            "threshold": svm_metrics_at_f1.threshold,
            "accuracy": svm_metrics_at_f1.accuracy,
            "precision": svm_metrics_at_f1.precision,
            "recall": svm_metrics_at_f1.recall,
            "f1": svm_metrics_at_f1.f1,
            "far": svm_metrics_at_f1.far,
            "frr": svm_metrics_at_f1.frr,
            "eer": svm_roc.eer,
            "eer_threshold": svm_roc.eer_threshold,
            "roc_auc": svm_roc.roc_auc,
            "pr_auc": svm_roc.pr_auc,
        }
    ]

    if ae_has_any:
        rows.append(
            {
                "model": "keyboard_ae",
                "metric_operating_point": "best_F1",
                "threshold": ae_metrics_at_f1.threshold,
                "accuracy": ae_metrics_at_f1.accuracy,
                "precision": ae_metrics_at_f1.precision,
                "recall": ae_metrics_at_f1.recall,
                "f1": ae_metrics_at_f1.f1,
                "far": ae_metrics_at_f1.far,
                "frr": ae_metrics_at_f1.frr,
                "eer": ae_roc.eer,
                "eer_threshold": ae_roc.eer_threshold,
                "roc_auc": ae_roc.roc_auc,
                "pr_auc": ae_roc.pr_auc,
            }
        )

    global_df = pd.DataFrame(rows)
    global_df.to_csv(
        os.path.join(RESULTS_DIR, "keyboard_models_global.csv"), index=False
    )

    # ---------- PLOTS ----------
    # Figure 4.1 – ROC AE vs SVM
    plt.figure()
    plt.plot(svm_roc.fpr, svm_roc.tpr, label=f"SVM (AUC={svm_roc.roc_auc:.3f})")
    if ae_has_any and ae_roc is not None:
        plt.plot(ae_roc.fpr, ae_roc.tpr, label=f"AE (AUC={ae_roc.roc_auc:.3f})")
    plt.xlabel("False Positive Rate (FAR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Keyboard ROC – AE vs SVM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_keyboard_roc_ae_vs_svm.png"))
    plt.close()

    # PR curve
    plt.figure()
    plt.plot(
        svm_roc.recall_curve,
        svm_roc.precision_curve,
        label=f"SVM (AUC={svm_roc.pr_auc:.3f})",
    )
    if ae_has_any and ae_roc is not None:
        plt.plot(
            ae_roc.recall_curve,
            ae_roc.precision_curve,
            label=f"AE (AUC={ae_roc.pr_auc:.3f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Keyboard PR – AE vs SVM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_keyboard_pr_ae_vs_svm.png"))
    plt.close()

    # Figure 4.2 – per-user SVM F1 bar chart
    per_user_sorted = per_user_df.sort_values("f1", ascending=False)
    plt.figure(figsize=(10, 4))
    plt.bar(per_user_sorted["user_id"], per_user_sorted["f1"])
    plt.xticks(rotation=90)
    plt.xlabel("User ID")
    plt.ylabel("F1-score")
    plt.title("Per-user Keyboard SVM F1-score")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "fig_keyboard_svm_per_user_f1.png")
    )
    plt.close()

    print("Keyboard evaluation complete. Tables & figures written to:", RESULTS_DIR)


if __name__ == "__main__":
    eval_keyboard_models()
