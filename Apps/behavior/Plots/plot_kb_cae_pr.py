"""Plot Precision-Recall curve for keyboard CAE trust.

Input:
  - artifacts/plots/kb_cae_scores.csv

Outputs:
  - artifacts/plots/kb_cae_pr_curve.csv
  - artifacts/plots/kb_cae_pr_curve.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Apps.behavior.ae_conditional import ARTIFACTS_DIR
from Apps.behavior.Plots.metrics_utils import compute_precision_recall


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
        description="Plot Precision-Recall curve for keyboard CAE trust.",
    )
    parser.add_argument(
        "--scores",
        type=str,
        default=str(DEFAULT_SCORES_CSV),
        help="Path to kb_cae_scores.csv.",
    )
    args = parser.parse_args()

    y_true, scores = _load_scores(Path(args.scores))

    precision, recall, thresh, pr_area = compute_precision_recall(y_true, scores)

    # Save curve data
    out_csv = PLOTS_DIR / "kb_cae_pr_curve.csv"
    df_curve = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
        }
    )
    df_curve.to_csv(out_csv, index=False)

    # Plot
    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (area ≈ {pr_area:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Keyboard CAE: Precision-Recall Curve (Trust Scores)")
    plt.grid(True)
    plt.legend()

    out_png = PLOTS_DIR / "kb_cae_pr_curve.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[OK] PR curve CSV -> {out_csv}")
    print(f"[OK] PR plot PNG  -> {out_png}")
    print(f"[INFO] Approx. area under PR curve ≈ {pr_area:.4f}")


if __name__ == "__main__":
    main()
