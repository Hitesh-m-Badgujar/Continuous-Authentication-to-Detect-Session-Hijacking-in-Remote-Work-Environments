# Apps/behavior/Plots/face/eval_face_scores.py
"""
Offline evaluation for the face module.

Inputs:
  - Data/face_scores.csv

Expected columns in face_scores.csv:
  - face_match : similarity score in [0, 1], higher = more similar
  - liveness   : liveness score in [0, 1]
  - label or is_genuine : 1 for genuine, 0 for impostor

We compute:
  - face_trust = face_match * liveness  (same as runtime fuse_face)
  - FAR / FRR vs threshold for face_trust
  - ROC curve (TPR vs FPR)
  - Histograms of genuine vs impostor scores
  - EER + AUC

Outputs (into artifacts/face):
  - face_scores_with_trust.csv
  - face_metrics.csv
  - face_far_frr.png
  - face_roc.png
  - face_hist.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
# H1/Apps/behavior/Plots/face/eval_face_scores.py
# parents[0] = .../Plots/face
# parents[1] = .../behavior/Plots
# parents[2] = .../behavior
# parents[3] = .../Apps
# parents[4] = .../H1  <-- project root
BASE_DIR = THIS_FILE.parents[4]

DATA_DIR = BASE_DIR / "Data"
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "face"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FACE_CSV = DATA_DIR / "face_scores.csv"


# ---------------------------------------------------------------------
# Load + preprocess scores
# ---------------------------------------------------------------------

def load_face_scores() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load face_scores.csv and return:
      - trust_scores: face_match * liveness
      - labels      : 1 = genuine, 0 = impostor
      - df          : original df with an extra 'face_trust' column
    """
    if not FACE_CSV.is_file():
        raise SystemExit(f"[FATAL] Face scores CSV not found: {FACE_CSV}")

    print(f"[INFO] Using face scores CSV: {FACE_CSV}")
    df = pd.read_csv(FACE_CSV)

    # Check columns
    missing = [c for c in ("face_match", "liveness") if c not in df.columns]
    if missing:
        raise SystemExit(f"[FATAL] face_scores.csv missing columns: {missing}")

    # Label column can be "label" or "is_genuine"
    if "label" in df.columns:
        labels_raw = df["label"].to_numpy()
    elif "is_genuine" in df.columns:
        labels_raw = df["is_genuine"].to_numpy()
    else:
        raise SystemExit(
            "[FATAL] face_scores.csv must contain 'label' or 'is_genuine' "
            "(1 = genuine, 0 = impostor)."
        )

    # Normalize labels to {0,1}
    labels = np.asarray(labels_raw, dtype=int)
    labels = (labels > 0).astype(int)

    face_match = np.clip(df["face_match"].to_numpy(dtype=float), 0.0, 1.0)
    liveness = np.clip(df["liveness"].to_numpy(dtype=float), 0.0, 1.0)

    # Same as runtime fuse_face: fm * lv
    face_trust = face_match * liveness
    face_trust = np.clip(face_trust, 0.0, 1.0)

    df = df.copy()
    df["face_trust"] = face_trust

    return face_trust, labels, df


# ---------------------------------------------------------------------
# FAR/FRR, ROC, EER
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

    fars = np.array(fars)
    frrs = np.array(frrs)
    tprs = np.array(tprs)
    fprs = np.array(fprs)

    # EER = point where FAR and FRR are closest
    diff = np.abs(fars - frrs)
    idx_eer = int(np.argmin(diff))
    eer = float((fars[idx_eer] + frrs[idx_eer]) / 2.0)
    eer_thr = float(thresholds[idx_eer])

    # ROC AUC – treat labels 1 (genuine) vs 0 (impostor)
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
# Plotting
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
    plt.axhline(eer, linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("Face – FAR / FRR vs threshold")
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
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Face – ROC curve")
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
    plt.xlabel("Face trust score")
    plt.ylabel("Density")
    plt.title("Face – score distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("[INFO] Loading face scores…")
    scores, labels, df = load_face_scores()

    print("[INFO] Computing FAR/FRR, ROC, EER…")
    metrics = compute_far_frr(scores, labels)

    # Save extended scores CSV
    out_scores = ARTIFACTS_DIR / "face_scores_with_trust.csv"
    df.to_csv(out_scores, index=False)
    print(f"[INFO] Saved scores -> {out_scores}")

    # Save metrics summary
    metrics_summary = {
        "EER": [metrics["EER"]],
        "EER_threshold": [metrics["EER_threshold"]],
        "AUC": [metrics["AUC"]],
        "n_samples": [len(scores)],
        "n_genuine": [int((labels == 1).sum())],
        "n_impostor": [int((labels == 0).sum())],
    }
    out_metrics = ARTIFACTS_DIR / "face_metrics.csv"
    pd.DataFrame(metrics_summary).to_csv(out_metrics, index=False)
    print(f"[INFO] Saved metrics -> {out_metrics}")

    # Plots
    far_frr_path = ARTIFACTS_DIR / "face_far_frr.png"
    roc_path = ARTIFACTS_DIR / "face_roc.png"
    hist_path = ARTIFACTS_DIR / "face_hist.png"

    print("[INFO] Plotting FAR/FRR curve…")
    plot_far_frr(metrics, far_frr_path)
    print(f"[INFO] Saved -> {far_frr_path}")

    print("[INFO] Plotting ROC curve…")
    plot_roc(metrics, roc_path)
    print(f"[INFO] Saved -> {roc_path}")

    print("[INFO] Plotting histograms…")
    plot_hist(scores, labels, hist_path)
    print(f"[INFO] Saved -> {hist_path}")

    print("[DONE] Face evaluation complete.")


if __name__ == "__main__":
    main()
