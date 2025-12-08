"""Plot FAR/FRR vs threshold for keyboard CAE trust.

Input:
  - artifacts/plots/kb_cae_scores.csv (from build_kb_cae_scores.py)

Outputs (in artifacts/plots):
  - kb_cae_far_frr_curve.csv
  - kb_cae_far_frr_vs_threshold.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Apps.behavior.ae_conditional import ARTIFACTS_DIR
from Apps.behavior.Plots.metrics_utils import compute_far_frr, compute_eer


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
        description="Plot FAR/FRR vs threshold for keyboard CAE trust.",
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

    # Save CSV with the curve
    out_csv = PLOTS_DIR / "kb_cae_far_frr_curve.csv"
    df_curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "FAR": fars,
            "FRR": frrs,
        }
    )
    df_curve.to_csv(out_csv, index=False)

    # Plot
    plt.figure()
    plt.plot(thresholds, fars, label="FAR")
    plt.plot(thresholds, frrs, label="FRR")

    # Mark EER point
    plt.scatter(
        [eer_threshold],
        [eer],
        marker="o",
        label=f"EER ≈ {eer:.3f} at τ ≈ {eer_threshold:.3f}",
    )

    plt.xlabel("Trust threshold (τ)")
    plt.ylabel("Error rate")
    plt.title("Keyboard CAE: FAR / FRR vs Trust Threshold")
    plt.grid(True)
    plt.legend()

    out_png = PLOTS_DIR / "kb_cae_far_frr_vs_threshold.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[OK] FAR/FRR curve CSV -> {out_csv}")
    print(f"[OK] FAR/FRR plot PNG  -> {out_png}")
    print(f"[INFO] EER ≈ {eer:.4f} at threshold ≈ {eer_threshold:.4f}")


if __name__ == "__main__":
    main()
