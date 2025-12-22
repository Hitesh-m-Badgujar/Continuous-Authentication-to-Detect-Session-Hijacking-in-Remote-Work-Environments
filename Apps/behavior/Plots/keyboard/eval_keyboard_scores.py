# Apps/behavior/Plots/keyboard/eval_keyboard_scores.py
"""
Offline evaluation for the keyboard model (SVM-based RuntimeScorer).

Outputs (into artifacts/keyboard):

  - kb_scores.csv
  - kb_metrics.csv
  - kb_far_frr.png
  - kb_roc.png
  - kb_hist.png

Definitions:

  - trust score  = RuntimeScorer's trust (or max_proba if exposed)
  - genuine      = sample where predicted user == true user
  - impostor     = sample where predicted user != true user

We DO NOT touch model files directly. Instead we reuse
Apps.behavior.ae_conditional.RuntimeScorer – the same thing the
real-time monitor uses – so this analysis is consistent with your UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------
# Fix sys.path so "Apps.behavior" can be imported when running as a script
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
# H1/Apps/behavior/Plots/keyboard/eval_keyboard_scores.py
# parents[0] = .../Plots/keyboard
# parents[1] = .../behavior/Plots
# parents[2] = .../behavior
# parents[3] = .../Apps
# parents[4] = .../H1  <-- project root
BASE_DIR = THIS_FILE.parents[4]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Now we can import your project modules
from Apps.behavior import ae_conditional  # type: ignore[attr-defined]

FEATURE_COLS = list(ae_conditional.FEATURE_COLS)
RuntimeScorer = ae_conditional.RuntimeScorer

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATA_DIR = BASE_DIR / "Data"
KB_CSV = DATA_DIR / "kb_cmu_windows.csv"

ARTIFACTS_DIR = BASE_DIR / "artifacts" / "keyboard"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Step 1 – load data + model and compute scores
# ---------------------------------------------------------------------

def _get_label_column(df: pd.DataFrame) -> str:
    """Find the column that contains the user ID / subject label."""
    for col in ("user_id", "subject", "user"):
        if col in df.columns:
            return col
    raise SystemExit(
        "[FATAL] Could not find user label column in keyboard CSV. "
        "Expected one of: user_id, subject, user."
    )


def _batch_scores_with_model(scorer: RuntimeScorer, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Try to do fast batch scoring via underlying SVM if available.
    Falls back to calling scorer.score_global row-by-row.
    """
    # Best case: RuntimeScorer exposes scaler + model with predict_proba
    scaler = getattr(scorer, "scaler", None)
    model = getattr(scorer, "model", None)

    if scaler is not None and model is not None and hasattr(model, "predict_proba"):
        X_scaled = scaler.transform(X)
        proba = model.predict_proba(X_scaled)
        proba_max = proba.max(axis=1)
        classes = model.classes_.astype(str)
        idx = np.argmax(proba, axis=1)
        y_pred = classes[idx]
        return proba_max.astype(float), y_pred.astype(str)

    # Fallback: call scorer.score_global for each row (slower but safe)
    scores = []
    preds = []
    for row in X:
        out = scorer.score_global(row.reshape(1, -1))
        # RuntimeScorer for SVM should expose trust + pred_user + prob
        trust = out.get("trust", out.get("prob", 0.0))
        pred_user = out.get("pred_user", "")
        scores.append(float(trust))
        preds.append(str(pred_user))

    return np.asarray(scores, dtype=float), np.asarray(preds, dtype=str)


def load_keyboard_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load keyboard windows + runtime model, compute trust scores and correctness labels.

    Returns:
      df        : original dataframe
      scores    : trust scores for all samples, shape (N,)
      labels    : 1 for genuine, 0 for impostor, shape (N,)
      proba_max : same as scores (for compatibility), shape (N,)
      y_true    : true user labels as strings, shape (N,)
    """
    if not KB_CSV.is_file():
        raise SystemExit(f"[FATAL] Keyboard CSV not found: {KB_CSV}")

    print(f"[INFO] Using keyboard CSV: {KB_CSV}")
    df = pd.read_csv(KB_CSV)

    label_col = _get_label_column(df)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[FATAL] Keyboard CSV missing feature columns: {missing}")

    df[label_col] = df[label_col].astype(str).str.strip()
    df = df.dropna(subset=FEATURE_COLS).copy()

    # Instantiate the same scorer used by the runtime monitor
    print("[INFO] Initialising RuntimeScorer from ae_conditional…")
    scorer = RuntimeScorer()  # uses internal model_dir config

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y_true = df[label_col].astype(str).to_numpy()

    print("[INFO] Computing trust scores + predicted users…")
    scores, y_pred = _batch_scores_with_model(scorer, X)

    # Genuine vs impostor labels
    correct = (y_pred == y_true)
    labels = np.zeros_like(scores, dtype=int)
    labels[correct] = 1  # 1 = genuine, 0 = impostor

    return df, scores, labels, scores, y_true  # proba_max == scores


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
    plt.title("Keyboard – FAR / FRR vs threshold")
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
    plt.title("Keyboard – ROC curve")
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
    plt.xlabel("Trust score")
    plt.ylabel("Density")
    plt.title("Keyboard – score distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def main() -> None:
    print("[INFO] Loading keyboard data + RuntimeScorer model…")
    df, scores, labels, proba_max, y_true = load_keyboard_data()

    print("[INFO] Computing FAR/FRR, ROC, EER…")
    metrics = compute_far_frr(scores, labels)

    # Save raw scores for later analysis / report
    scores_df = pd.DataFrame({
        "trust_score": scores,
        "label": labels,  # 1 = genuine, 0 = impostor
        "user_id": y_true,
    })
    scores_csv_path = ARTIFACTS_DIR / "kb_scores.csv"
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
    metrics_csv_path = ARTIFACTS_DIR / "kb_metrics.csv"
    pd.DataFrame(metrics_summary).to_csv(metrics_csv_path, index=False)
    print(f"[INFO] Saved metrics -> {metrics_csv_path}")

    # Plots
    far_frr_path = ARTIFACTS_DIR / "kb_far_frr.png"
    roc_path = ARTIFACTS_DIR / "kb_roc.png"
    hist_path = ARTIFACTS_DIR / "kb_hist.png"

    print("[INFO] Plotting FAR/FRR curve…")
    plot_far_frr(metrics, far_frr_path)
    print(f"[INFO] Saved -> {far_frr_path}")

    print("[INFO] Plotting ROC curve…")
    plot_roc(metrics, roc_path)
    print(f"[INFO] Saved -> {roc_path}")

    print("[INFO] Plotting histograms…")
    plot_hist(scores, labels, hist_path)
    print(f"[INFO] Saved -> {hist_path}")

    print("[DONE] Keyboard evaluation complete.")


if __name__ == "__main__":
    main()
