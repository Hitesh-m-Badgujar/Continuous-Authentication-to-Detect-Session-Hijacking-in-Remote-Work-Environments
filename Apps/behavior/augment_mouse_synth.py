# Apps/behavior/augment_mouse_synth.py
"""
Generate synthetic mouse window data to augment the training set.

Inputs:
  - Data/mouse_windows_train.csv

Outputs:
  - Data/mouse_windows_train_augmented.csv

For each user_id:
  - Take their real windows.
  - Generate synthetic windows by adding Gaussian noise per feature:
        x_syn = x_real + N(0, noise_scale * std_feature)
  - Clamp obvious non-negative features to >= 0.

This is *legit* data augmentation. It may or may not push test accuracy
above 85% – that depends on the true separability of the problem.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"

INPUT_CSV = DATA_DIR / "mouse_windows_train.csv"
OUTPUT_CSV = DATA_DIR / "mouse_windows_train_augmented.csv"

LABEL_COL = "user_id"

# Must match your existing feature schema
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

# How many synthetic samples per real sample (e.g. 1.0 → double data)
AUG_FACTOR = 1.0

# Noise scale relative to feature std dev (tune if needed)
NOISE_SCALE = 0.30


def _ensure_non_negative(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=0.0)


def main() -> None:
    print(f"[INFO] Loading training data from: {INPUT_CSV}")
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"mouse_windows_train.csv not found at {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Training CSV missing feature columns: {missing}")

    if LABEL_COL not in df.columns:
        raise RuntimeError(f"Training CSV missing label column '{LABEL_COL}'")

    print(f"[INFO] Original rows: {len(df)}")

    all_rows = [df]  # list of DataFrames (original + synthetic)

    # Generate per-user synthetic data
    for user_id, df_user in df.groupby(LABEL_COL):
        X = df_user[FEATURE_COLS].to_numpy(dtype=float)
        n_real = X.shape[0]
        if n_real < 5:
            print(f"[WARN] Skipping user {user_id} (too few samples: {n_real})")
            continue

        n_synth = int(round(n_real * AUG_FACTOR))
        if n_synth <= 0:
            continue

        print(f"[INFO] Generating {n_synth} synthetic windows for {user_id} (real={n_real})")

        # Per-feature mean and std
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)

        # Avoid zero std (no variation) – use small epsilon
        sigma = np.where(sigma < 1e-8, 1e-8, sigma)

        # Sample index to pick real base rows
        base_idx = np.random.randint(0, n_real, size=n_synth)
        X_base = X[base_idx]

        noise = np.random.normal(loc=0.0, scale=NOISE_SCALE, size=X_base.shape)
        X_syn = X_base + noise * sigma  # broadcast sigma

        # Build synthetic DataFrame
        df_syn = pd.DataFrame(X_syn, columns=FEATURE_COLS)
        df_syn[LABEL_COL] = str(user_id)

        # Copy over any non-feature columns with defaults
        for col in df.columns:
            if col in FEATURE_COLS or col == LABEL_COL:
                continue
            # e.g., 'file', 'start_idx', 'end_idx' – we can mark as synthetic
            if col == "file":
                df_syn[col] = "synthetic"
            elif col in ("start_idx", "end_idx"):
                df_syn[col] = -1
            else:
                df_syn[col] = df_user[col].iloc[0]

        # Clamp non-negative features
        nonneg_cols = [
            "dur_ms",
            "n_points",
            "path_len",
            "straight_len",
            "mean_speed",
            "p95_speed",
            "max_speed",
            "mean_acc",
            "p95_acc",
            "max_acc",
            "mean_jerk",
            "p95_jerk",
            "max_jerk",
            "abs_dx",
            "abs_dy",
            "bbox_w",
            "bbox_h",
            "bbox_area",
            "direction_changes",
            "pause_ratio_20ms",
        ]
        _ensure_non_negative(df_syn, nonneg_cols)

        all_rows.append(df_syn)

    df_aug = pd.concat(all_rows, ignore_index=True)
    print(f"[INFO] Augmented rows (real + synthetic): {len(df_aug)}")

    df_aug.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved augmented dataset to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
