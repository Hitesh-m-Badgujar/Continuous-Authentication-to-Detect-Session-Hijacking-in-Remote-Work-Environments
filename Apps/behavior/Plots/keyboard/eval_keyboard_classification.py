# Apps/behavior/Plots/keyboard/eval_keyboard_classification.py
"""
Offline classification metrics for keyboard SVM on CMU windows.

Computes:
  - Overall accuracy
  - Precision / recall / F1-score (macro + weighted)
  - Full classification report
  - Confusion matrix heatmap (saved as PNG)
  - Per-user performance table (saved as CSV)

Inputs:
  - Data/kb_cmu_windows.csv

Outputs (created under Artifacts/plots/keyboard):
  - kb_confusion_matrix.png
  - kb_confusion_matrix_normalized.png
  - kb_per_user_metrics.csv
  - kb_classification_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------
# Path setup so "from Apps.behavior import ae_conditional" works
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
# .../H1/Apps/behavior/Plots/keyboard/eval_keyboard_classification.py
# parents[0]=keyboard, [1]=Plots, [2]=behavior, [3]=Apps, [4]=H1
BASE_DIR = THIS_FILE.parents[4]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Apps.behavior import ae_conditional  # type: ignore  # noqa: E402

DATA_DIR = BASE_DIR / "Data"
ARTIFACTS_DIR = BASE_DIR / "Artifacts" / "plots" / "keyboard"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

KB_CSV = DATA_DIR / "kb_cmu_windows.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_keyboard_df() -> pd.DataFrame:
    if not KB_CSV.is_file():
        raise SystemExit(f"[FATAL] Keyboard CSV not found: {KB_CSV}")

    df = pd.read_csv(KB_CSV)
    if "user_id" not in df.columns:
        raise SystemExit("[FATAL] Expected 'user_id' column in kb_cmu_windows.csv")

    missing = [c for c in ae_conditional.FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            "[FATAL] kb_cmu_windows.csv is missing feature columns: "
            + ", ".join(missing)
        )

    df = df.dropna(subset=ae_conditional.FEATURE_COLS).copy()
    df["user_id"] = df["user_id"].astype(str).str.strip()

    print(
        f"[INFO] Loaded {len(df)} keyboard windows for "
        f"{df['user_id'].nunique()} users from {KB_CSV}"
    )
    return df


def train_keyboard_svm(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    """
    Simple SVM training for offline evaluation.

    This does NOT touch your runtime model; it's just for metrics.
    """
    print("[INFO] Train/test split (stratified by user)…")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reasonable defaults; you can tweak if you like
    svm = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        decision_function_shape="ovr",
        probability=False,
    )

    print("[INFO] Training keyboard SVM for evaluation…")
    svm.fit(X_train_scaled, y_train)
    print("[INFO] Training complete.")

    return {
        "scaler": scaler,
        "model": svm,
        "X_test": X_test_scaled,
        "y_test": y_test,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    normalize: bool,
    out_path: Path,
) -> None:
    """
    Save confusion matrix heatmap (normal or normalized).
    """
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, cm_sum, where=cm_sum != 0)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(
        ticks=np.arange(len(labels)),
        labels=labels,
        rotation=90,
        fontsize=7,
    )
    plt.yticks(
        ticks=np.arange(len(labels)),
        labels=labels,
        fontsize=7,
    )

    plt.xlabel("Predicted user")
    plt.ylabel("True user")
    title = "Keyboard confusion matrix"
    if normalize:
        title += " (normalized)"
    plt.title(title)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved confusion matrix: {out_path}")


def compute_per_user_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    Per-user performance: for each user, how often are they correctly recognised.
    (Essentially per-class recall / accuracy.)
    """
    users = sorted(np.unique(y_true))
    rows = []

    for u in users:
        mask = (y_true == u)
        n = int(mask.sum())
        if n == 0:
            continue
        correct = int((y_true[mask] == y_pred[mask]).sum())
        acc = correct / n
        rows.append(
            {
                "user_id": u,
                "n_samples": n,
                "n_correct": correct,
                "user_accuracy": acc,
            }
        )

    df = pd.DataFrame(rows).sort_values("user_accuracy", ascending=False)
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    print("[INFO] Loading keyboard data…")
    df = load_keyboard_df()

    X = df[ae_conditional.FEATURE_COLS].to_numpy(dtype=float)
    y = df["user_id"].astype(str).to_numpy()

    eval_bundle = train_keyboard_svm(X, y)
    scaler = eval_bundle["scaler"]
    model = eval_bundle["model"]
    X_test = eval_bundle["X_test"]
    y_test = eval_bundle["y_test"]

    print("[INFO] Evaluating on held-out test set…")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Overall accuracy: {acc:.4f}")

    report = classification_report(
        y_test,
        y_pred,
        digits=4,
        output_dict=False,
    )
    print("\n[RESULT] Classification report:\n")
    print(report)

    # Save text report
    report_path = ARTIFACTS_DIR / "kb_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(f"Overall accuracy: {acc:.4f}\n\n")
        fh.write(report)
    print(f"[INFO] Saved classification report: {report_path}")

    # Confusion matrices
    labels = sorted(pd.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_path = ARTIFACTS_DIR / "kb_confusion_matrix.png"
    cm_norm_path = ARTIFACTS_DIR / "kb_confusion_matrix_normalized.png"

    plot_confusion_matrix(cm, labels, normalize=False, out_path=cm_path)
    plot_confusion_matrix(cm, labels, normalize=True, out_path=cm_norm_path)

    # Per-user metrics
    per_user_df = compute_per_user_metrics(y_test, y_pred)
    per_user_path = ARTIFACTS_DIR / "kb_per_user_metrics.csv"
    per_user_df.to_csv(per_user_path, index=False)
    print(f"[INFO] Saved per-user metrics: {per_user_path}")

    print("\n[DONE] Keyboard metrics ready.")
    print("  - Accuracy / precision / recall / F1 in kb_classification_report.txt")
    print("  - Confusion matrices PNGs in Artifacts/plots/keyboard/")
    print("  - Per-user metrics CSV in Artifacts/plots/keyboard/")


if __name__ == "__main__":
    main()
