"""Confusion matrix + basic metrics at EER threshold for keyboard CAE trust.

Input:
  - artifacts/plots/kb_cae_scores.csv

Outputs:
  - artifacts/plots/kb_cae_confusion_eer.csv
  - artifacts/plots/kb_cae_confusion_eer.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Apps.behavior.ae_conditional import ARTIFACTS_DIR
from Apps.behavior.Plots.metrics_utils import (
    compute_far_frr,
    compute_eer,
    compute_confusion_at_threshold,
    compute_basic_metrics,
)


PLOTS_DIR = ARTIFACTS_DIR / "plots"
DEFAULT_SCORES_CSV = PLOTS_DIR / "kb_cae_scores.csv"


def _load_scores(csv_path: Path):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise SystemExit(
            f"Scores CSV not found: {csv_path}\nRun build_kb_cae_scores.py first."
        )

    df = pd.read_csv(csv_path)
    if "trust" not in df.columns or "label" not in df.columns:
        raise SystemExit("kb_cae_scores.csv must have 'trust' and 'label' columns.")

    y_true = df["label"].to_numpy(dtype=int)
    scores = df["trust"].to_numpy(dtype=float)
    return y_true, scores


def main():
    parser = argparse.ArgumentParser(
        description="Confusion matrix + metrics at EER threshold for keyboard CAE.",
    )
    parser.add_argument(
        "--scores",
        type=str,
        default=str(DEFAULT_SCORES_CSV),
        help="Path to kb_cae_scores.csv.",
    )
    args = parser.parse_args()

    y_true, scores = _load_scores(Path(args.scores))

    thresholds = np.linspace(0.0, 1.0, 201)
    thresholds, fars, frrs = compute_far_frr(y_true, scores, thresholds=thresholds)
    eer, eer_threshold = compute_eer(fars, frrs, thresholds)

    tp, fp, tn, fn = compute_confusion_at_threshold(y_true, scores, eer_threshold)
    metrics = compute_basic_metrics(tp, fp, tn, fn)

    # Save confusion + metrics table
    df = pd.DataFrame(
        {
            "TP": [tp],
            "FP": [fp],
            "TN": [tn],
            "FN": [fn],
            "EER": [eer],
            "EER_threshold": [eer_threshold],
            "accuracy": [metrics["accuracy"]],
            "precision": [metrics["precision"]],
            "recall": [metrics["recall"]],
            "f1": [metrics["f1"]],
        }
    )
    out_csv = PLOTS_DIR / "kb_cae_confusion_eer.csv"
    df.to_csv(out_csv, index=False)

    # Simple 2x2 heatmap
    mat = np.array([[tp, fp], [fn, tn]], dtype=float)
    labels = np.array([["TP", "FP"], ["FN", "TN"]])

    plt.figure()
    plt.imshow(mat, interpolation="nearest")
    plt.title("Keyboard CAE Confusion Matrix at EER Threshold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred Genuine", "Pred Impostor"])
    plt.yticks(tick_marks, ["True Genuine", "True Impostor"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{labels[i, j]} = {int(mat[i, j])}",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    out_png = PLOTS_DIR / "kb_cae_confusion_eer.png"
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[OK] Confusion matrix CSV -> {out_csv}")
    print(f"[OK] Confusion matrix PNG -> {out_png}")
    print(
        f"[INFO] EER ≈ {eer:.4f} at threshold ≈ {eer_threshold:.4f} | "
        f"Accuracy ≈ {metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
