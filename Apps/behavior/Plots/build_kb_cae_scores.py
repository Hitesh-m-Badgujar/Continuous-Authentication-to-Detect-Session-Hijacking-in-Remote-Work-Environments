"""Build per-window keyboard CAE trust scores for genuine vs impostor.

This script:
  - loads keyboard feature windows from Data/kb_cmu_windows.csv
  - uses Apps.behavior.ae_conditional.RuntimeScorer to compute residuals + trust
  - for each user:
      * treats that user's windows as GENUINE
      * treats all other users' windows as IMPOSTOR
      * samples a balanced subset (min_genuine, max_impostor)
  - writes a flat CSV with one row per sampled window:

      artifacts/plots/kb_cae_scores.csv

    columns:
      user_id, session_id, window_id, trust, residual, label
      (label = 1 for genuine, 0 for impostor)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from Apps.behavior.ae_conditional import (
    BASE_DIR,
    DATA_DIR,
    ARTIFACTS_DIR,
    FEATURE_COLS,
    RuntimeScorer,
    _compute_residuals,
    DEFAULT_WINDOWS_CSV,
)


PLOTS_DIR = ARTIFACTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_windows(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise SystemExit(f"Keyboard windows CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"{csv_path.name} exists but is empty.")

    if "user_id" not in df.columns:
        raise SystemExit("Expected 'user_id' column in keyboard windows CSV.")

    # Normalise user_id and drop rows with missing features
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df = df.dropna(subset=FEATURE_COLS).copy()

    # Optional: sort for stability (not strictly required)
    sort_cols: List[str] = [c for c in ["session_id", "window_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def build_kb_cae_scores(
    csv_path: Path = DEFAULT_WINDOWS_CSV,
    min_genuine: int = 200,
    max_impostor: int | None = None,
) -> Path:
    """Main routine to build kb_cae_scores.csv.

    Returns the path to the generated CSV.
    """
    df = _load_windows(csv_path)

    # Initialise runtime scorer once
    scorer = RuntimeScorer()

    # Compute residuals + trust for all rows once to avoid re-predicting
    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    residuals = _compute_residuals(X_all, scorer.scaler, scorer.model)
    trusts = np.array(
        [scorer._trust_from_residual(float(r)) for r in residuals],
        dtype=np.float32,
    )

    df["residual"] = residuals
    df["trust"] = trusts

    rows: List[Dict[str, Any]] = []

    users = sorted(df["user_id"].unique())
    for user in users:
        df_g = df[df["user_id"] == user]
        df_i = df[df["user_id"] != user]

        if len(df_g) < min_genuine or len(df_i) < min_genuine:
            print(
                f"[WARN] Skipping user {user}: "
                f"n_genuine={len(df_g)}, n_impostor={len(df_i)} (need >= {min_genuine} of each)"
            )
            continue

        n_g = min(len(df_g), min_genuine)
        n_i = min(len(df_i), max_impostor or n_g)

        df_g_sample = df_g.sample(n=n_g, random_state=123, replace=False)
        df_i_sample = df_i.sample(n=n_i, random_state=123, replace=False)

        for _, row in df_g_sample.iterrows():
            rows.append(
                {
                    "user_id": row["user_id"],
                    "session_id": row.get("session_id", None),
                    "window_id": row.get("window_id", None),
                    "trust": float(row["trust"]),
                    "residual": float(row["residual"]),
                    "label": 1,
                }
            )

        for _, row in df_i_sample.iterrows():
            rows.append(
                {
                    "user_id": row["user_id"],
                    "session_id": row.get("session_id", None),
                    "window_id": row.get("window_id", None),
                    "trust": float(row["trust"]),
                    "residual": float(row["residual"]),
                    "label": 0,
                }
            )

    if not rows:
        raise SystemExit("No rows collected; check data and thresholds.")

    out_df = pd.DataFrame(rows)
    out_path = PLOTS_DIR / "kb_cae_scores.csv"
    out_df.to_csv(out_path, index=False)

    print(f"[OK] Wrote keyboard CAE scores to: {out_path}")
    print(
        f"[INFO] Total samples: {len(out_df)} | "
        f"genuine={int((out_df['label'] == 1).sum())} | "
        f"impostor={int((out_df['label'] == 0).sum())}"
    )

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Build per-window keyboard CAE trust scores for plotting.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_WINDOWS_CSV),
        help="Keyboard windows CSV (default: Data/kb_cmu_windows.csv).",
    )
    parser.add_argument(
        "--min-genuine",
        type=int,
        default=200,
        help="Minimum genuine windows per user.",
    )
    parser.add_argument(
        "--max-impostor",
        type=int,
        default=None,
        help="Maximum impostor windows per user (default: same as genuine).",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    min_genuine = int(args.min_genuine)
    max_impostor = int(args.max_impostor) if args.max_impostor is not None else None

    build_kb_cae_scores(
        csv_path=csv_path,
        min_genuine=min_genuine,
        max_impostor=max_impostor,
    )


if __name__ == "__main__":
    main()
