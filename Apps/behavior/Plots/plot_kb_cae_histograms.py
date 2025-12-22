"""Plot histograms of keyboard CAE trust scores for genuine vs impostor.

Input:
  - artifacts/plots/kb_cae_scores.csv

Output:
  - artifacts/plots/kb_cae_trust_histograms.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Apps.behavior.ae_conditional import ARTIFACTS_DIR


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
        description="Plot histograms of keyboard CAE trust scores.",
    )
    parser.add_argument(
        "--scores",
        type=str,
        default=str(DEFAULT_SCORES_CSV),
        help="Path to kb_cae_scores.csv.",
    )
    args = parser.parse_args()

    y_true, scores = _load_scores(Path(args.scores))

    genuine_scores = scores[y_true == 1]
    impostor_scores = scores[y_true == 0]

    if genuine_scores.size == 0 or impostor_scores.size == 0:
        raise SystemExit("Need both genuine (1) and impostor (0) samples.")

    all_scores = np.concatenate([genuine_scores, impostor_scores])
    bins = np.linspace(all_scores.min(), all_scores.max(), 40)

    plt.figure()
    plt.hist(
        genuine_scores,
        bins=bins,
        alpha=0.6,
        label="Genuine (label=1)",
        density=True,
    )
    plt.hist(
        impostor_scores,
        bins=bins,
        alpha=0.6,
        label="Impostor (label=0)",
        density=True,
    )

    plt.xlabel("Keyboard CAE trust score")
    plt.ylabel("Density")
    plt.title("Keyboard CAE: Trust Score Distributions")
    plt.grid(True)
    plt.legend()

    out_png = PLOTS_DIR / "kb_cae_trust_histograms.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[OK] Histogram plot PNG -> {out_png}")


if __name__ == "__main__":
    main()
