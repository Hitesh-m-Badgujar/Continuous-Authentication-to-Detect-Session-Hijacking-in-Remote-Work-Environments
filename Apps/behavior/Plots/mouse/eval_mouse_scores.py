# Apps/behavior/Plots/mouse/eval_mouse_scores.py
"""
Offline evaluation for the mouse SVM classifier.

Outputs (into artifacts/mouse):

  - mouse_scores.csv
  - mouse_metrics.csv
  - mouse_far_frr.png
  - mouse_roc.png
  - mouse_hist.png

Definitions:

  - trust score  = max predicted probability (same as runtime)
  - genuine      = sample where predicted user == true user
  - impostor     = sample where predicted user != true user

This is not perfect biometric evaluation, but it's consistent with how the
runtime engine uses the mouse SVM, and it's good enough for your report
(FAR/FRR, ROC, histograms, EER).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load as joblib_load
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

# This file is: H1/Apps/behavior/Plots/mouse/eval_mouse_scores.py
#   parents[0] = .../Plots/mouse
#   parents[1] = .../behavior/Plots
#   parents[2] = .../behavior
#   parents[3] = .../Apps
#   parents[4] = .../H1  <-- project root
BASE_DIR = Path(__file__).resolve().parents[4]

DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models" / "mouse"
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "mouse"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Prefer augmented CSV if it exists, else fall back to original
TRAIN_CSV_AUG = DATA_DIR / "mouse_windows_train_augmented.csv"
TRAIN_CSV_ORIG = DATA_DIR / "mouse_windows_train.csv"

if TRAIN_CSV_AUG.is_file():
    TRAIN_CSV = TRAIN_CSV_AUG
else:
    TRAIN_CSV = TRAIN_CSV_ORIG

SCALER_PATH = MODELS_DIR / "mouse_scaler.joblib"
MODEL_PATH = MODELS_DIR / "mouse_svm.joblib"

# Must match your training script
FEATURE_COLS = [
    "dur_ms",
    "n_points",
    "path_len",
    "straight_len",
    "straightness",
    "mean_speed",
    "p95_speed",
    "max_speed",
    "mean_acc",
    "p95_acc",
    "max_acc",
    "mean_jerk",
    "p95_jerk",
    "max_jerk",
    "dx",
    "dy",
    "abs_dx",
    "abs_dy",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "direction_changes",
    "pause_ratio_20ms",
]


# ---------------------------------------------------------------------
# Step 1 – load data + model and compute scores
# ---------------------------------------------------------------------

def load_mouse_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load mouse windows + trained model, compute trust scores and correctness labels.

    Returns:
      df        : original dataframe
      scores    : trust scores (max_proba) for all samples, shape (N,)
      labels    : 1 for genuine, 0 for impostor, shape (N,)
      proba_max : max predicted probability for each sample, shape (N,)
      y_true    : true user_id labels as strings, shape (N,)
    """
    if not TRAIN_CSV.is_file():
        raise SystemExit(
            "[FATAL] Mouse train CSV not found.\n"
            f"Tried:\n  {TRAIN_CSV_AUG}\n  {TRAIN_CSV_ORIG}\n"
        )

    print(f"[INFO] Using mouse train CSV: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)
    if "user_id" not in df.columns:
        raise SystemExit("[FATAL] Expected 'user_id' column in mouse CSV")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[FATAL] Mouse CSV missing feature columns: {missing}")

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df = df.dropna(subset=FEATURE_COLS).copy()

    if not SCALER_PATH.is_file() or not MODEL_PATH.is_file():
        raise SystemExit(
            f"[FATAL] Mouse model/scaler not found.\n"
            f"  scaler: {SCALER_PATH}\n"
            f"  model : {MODEL_PATH}\n"
            f"Train the model first with train_mouse_svm.py."
        )

    scaler = joblib_load(SCALER_PATH)
    model = joblib_load(MODEL_PATH)

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y_true = df["user_id"].astype(str).to_numpy()

    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)  # shape (N, n_classes)
    proba_max = proba.max(axis=1)
    y_pred_idx = np.argmax(proba, axis=1)
    classes = model.classes_.astype(str)
    y_pred = classes[y_pred_idx]

    correct = (y_pred == y_true)
    scores = proba_max.astype(float)

    labels = np.zeros_like(scores, dtype=int)
    labels[correct] = 1  # 1 = genuine, 0 = impostor

    return df, scores, labels, proba_max, y_true


# ---------------------------------------------------------------------
# Step 2 – compute FAR/FRR, ROC, EER
# ---------------------------------------------------------------------

def compute_far_frr(scores: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Compute FAR / FRR vs threshold, ROC-like curve, AUC, EER.

    labels: 1 for genuine, 0 for impostor
    scores: higher = more genuine / trusted
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    thresholds = np.linspace(0.0, 1.0, 501)

    fars = []
    frrs = []
    tprs = []
    fprs = []

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    if genuine_scores.size == 0 or impostor_scores.size == 0:
        raise SystemExit("[FATAL] Need both genuine and impostor samples to compute FAR/FRR.")

    for thr in thresholds:
        # Genuine
        tp = np.sum(genuine_scores >= thr)
        fn = np.sum(genuine_scores < thr)

        # Impostor
        fp = np.sum(impostor_scores >= thr)
        tn = np.sum(impostor_scores < thr)

        fn_denom = tp + fn
        fp_denom = fp + tn

        frr = (fn / fn_denom) if fn_denom > 0 else 0.0
        far = (fp / fp_denom) if fp_denom > 0 else 0.0

        tpr = 1.0 - frr
        fpr = far

        frrs.append(frr)
        fars.append(far)
        tprs.append(tpr)
        fprs.append(fpr)

    thresholds = thresholds
    fars = np.array(fars)
    frrs = np.array(frrs)
    tprs = np.array(tprs)
    fprs = np.array(fprs)

    # EER = point where FAR and FRR are closest
    diff = np.abs(fars - frrs)
    idx_eer = int(np.argmin(diff))
    eer = float((fars[idx_eer] + frrs[idx_eer]) / 2.0)
    eer_thr = float(thresholds[idx_eer])

    # ROC AUC
    try:
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = float("nan")

    return {
        "thresholds": thresholds,
        "FAR": fars,
        "FRR": frrs,
        "TPR": tprs,
        "FPR": fprs,
        "EER": eer,
        "EER_threshold": eer_thr,
        "AUC": auc,
    }


# ---------------------------------------------------------------------
# Step 3 – plotting
# ---------------------------------------------------------------------

def plot_far_frr(metrics: Dict[str, Any], out_path: Path) -> None:
    thr = metrics["thresholds"]
    far = metrics["FAR"]
    frr = metrics["FRR"]
    eer = metrics["EER"]
    eer_thr = metrics["EER_threshold"]

    plt.figure()
    plt.plot(thr, far, label="FAR")
    plt.plot(thr, frr, label="FRR")
    plt.axvline(eer_thr, linestyle="--", label=f"EER@{eer_thr:.3f}")
    plt.axhline(eer, linestyle="--", color="grey")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("Mouse SVM – FAR / FRR vs threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc(metrics: Dict[str, Any], out_path: Path) -> None:
    fpr = metrics["FPR"]
    tpr = metrics["TPR"]
    auc = metrics["AUC"]

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Mouse SVM – ROC curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_hist(scores: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]

    plt.figure()
    plt.hist(genuine, bins=30, alpha=0.6, density=True, label="Genuine")
    plt.hist(impostor, bins=30, alpha=0.6, density=True, label="Impostor")
    plt.xlabel("Trust score (max_proba)")
    plt.ylabel("Density")
    plt.title("Mouse SVM – score distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def main() -> None:
    print("[INFO] Loading mouse data + model…")
    df, scores, labels, proba_max, y_true = load_mouse_data()

    print("[INFO] Computing FAR/FRR, ROC, EER…")
    metrics = compute_far_frr(scores, labels)

    # Save raw scores for later analysis / report
    scores_df = pd.DataFrame({
        "trust_score": scores,
        "label": labels,  # 1 = genuine, 0 = impostor
        "user_id": y_true,
    })
    scores_csv_path = ARTIFACTS_DIR / "mouse_scores.csv"
    scores_df.to_csv(scores_csv_path, index=False)
    print(f"[INFO] Saved scores -> {scores_csv_path}")

    # Save metrics summary
    metrics_summary = {
        "EER": [metrics["EER"]],
        "EER_threshold": [metrics["EER_threshold"]],
        "AUC": [metrics["AUC"]],
        "n_samples": [len(scores)],
        "n_genuine": [int((labels == 1).sum())],
        "n_impostor": [int((labels == 0).sum())],
    }
    metrics_csv_path = ARTIFACTS_DIR / "mouse_metrics.csv"
    pd.DataFrame(metrics_summary).to_csv(metrics_csv_path, index=False)
    print(f"[INFO] Saved metrics -> {metrics_csv_path}")

    # Plots
    far_frr_path = ARTIFACTS_DIR / "mouse_far_frr.png"
    roc_path = ARTIFACTS_DIR / "mouse_roc.png"
    hist_path = ARTIFACTS_DIR / "mouse_hist.png"

    print("[INFO] Plotting FAR/FRR curve…")
    plot_far_frr(metrics, far_frr_path)
    print(f"[INFO] Saved -> {far_frr_path}")

    print("[INFO] Plotting ROC curve…")
    plot_roc(metrics, roc_path)
    print(f"[INFO] Saved -> {roc_path}")

    print("[INFO] Plotting histograms…")
    plot_hist(scores, labels, hist_path)
    print(f"[INFO] Saved -> {hist_path}")

    print("[DONE] Mouse SVM evaluation complete.")


if __name__ == "__main__":
    main()
