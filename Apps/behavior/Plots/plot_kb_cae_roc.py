"""Plot ROC curve for keyboard CAE trust.

Input:
  - artifacts/plots/kb_cae_scores.csv

Outputs:
  - artifacts/plots/kb_cae_roc_curve.csv
  - artifacts/plots/kb_cae_roc_curve.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Apps.behavior.ae_conditional import ARTIFACTS_DIR
from Apps.behavior.Plots.metrics_utils import compute_roc


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
        description="Plot ROC curve for keyboard CAE trust.",
    )
    parser.add_argument(
        "--scores",
        type=str,
        default=str(DEFAULT_SCORES_CSV),
        help="Path to kb_cae_scores.csv.",
    )
    args = parser.parse_args()

    y_true, scores = _load_scores(Path(args.scores))

    fpr, tpr, thresh, roc_auc = compute_roc(y_true, scores)

    # Save curve data
    out_csv = PLOTS_DIR / "kb_cae_roc_curve.csv"
    df_curve = pd.DataFrame(
        {
            "FPR": fpr,
            "TPR": tpr,
            "threshold": thresh,
        }
    )
    df_curve.to_csv(out_csv, index=False)

    # Plot
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Keyboard CAE: ROC Curve (Trust Scores)")
    plt.grid(True)
    plt.legend()

    out_png = PLOTS_DIR / "kb_cae_roc_curve.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[OK] ROC curve CSV -> {out_csv}")
    print(f"[OK] ROC plot PNG  -> {out_png}")
    print(f"[INFO] AUC ≈ {roc_auc:.4f}")


if __name__ == "__main__":
    main()
