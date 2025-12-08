# Apps/behavior/train_mouse_svm.py
"""
Train a mouse SVM classifier on window-level features.

Inputs:
  - Data/mouse_windows_train.csv

Outputs:
  - Models/mouse/mouse_scaler.joblib
  - Models/mouse/mouse_svm.joblib
  - Models/mouse/mouse_meta.json

Aligned with Apps/behavior/eval_mouse.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models" / "mouse"

TRAIN_CSV = DATA_DIR / "mouse_windows_train_augmented.csv"

FEATURE_COLS: List[str] = [
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

MIN_PER_USER = 40
TEST_FRACTION = 0.25  # internal holdout for classification report


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_train_df() -> pd.DataFrame:
    if not TRAIN_CSV.is_file():
        raise SystemExit(f"Mouse train CSV not found: {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)
    if "user_id" not in df.columns:
        raise SystemExit("Expected 'user_id' column in mouse_windows_train.csv")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"mouse_windows_train.csv is missing feature columns: {missing}")

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df = df.dropna(subset=FEATURE_COLS).copy()

    vc = df["user_id"].value_counts()
    keep_users = vc[vc >= MIN_PER_USER].index
    df = df[df["user_id"].isin(keep_users)].copy()

    if df.empty:
        raise SystemExit("After MIN_PER_USER filtering, no users remain in mouse data.")

    print(
        f"[INFO] Loaded {len(df)} train windows for "
        f"{df['user_id'].nunique()} users from {TRAIN_CSV}"
    )
    return df


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_train_df()
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["user_id"].astype(str).to_numpy()

    # Train / test split for internal report (NOT the same as eval_mouse test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=42,
        stratify=y,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Wider hyperparameter search
    C_values = [1.0, 5.0, 10.0, 20.0, 50.0]
    gammas = ["scale", "auto", 0.01, 0.05, 0.1, 0.2]

    best_C = None
    best_gamma = None
    best_acc = -1.0

    print("[INFO] Tuning SVM hyperparameters on internal holdout...")
    for C in C_values:
        for gamma in gammas:
            svm = SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                decision_function_shape="ovr",
                class_weight="balanced",
                probability=True,  # needed later for trust score
            )
            svm.fit(X_train_scaled, y_train)
            acc = svm.score(X_test_scaled, y_test)
            print(f"  C={C:<5} gamma={str(gamma):<6} -> acc={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_C = C
                best_gamma = gamma

    print(
        f"[INFO] Best hyperparams: C={best_C}, gamma={best_gamma}, "
        f"internal_acc={best_acc:.4f}"
    )

    # Train final SVM on ALL training windows (train + test combined)
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X)
    svm_final = SVC(
        kernel="rbf",
        C=best_C,
        gamma=best_gamma,
        decision_function_shape="ovr",
        class_weight="balanced",
        probability=True,
    )
    print("[INFO] Training final mouse SVM on full train set...")
    svm_final.fit(X_all_scaled, y)
    print("[INFO] Final mouse SVM training done.")

    # Optional: classification report on the internal holdout with final model
    y_test_pred = svm_final.predict(scaler.transform(X_test))
    print("\n[INFO] Classification report on internal holdout:")
    print(classification_report(y_test, y_test_pred))

    # Save artifacts
    scaler_path = MODELS_DIR / "mouse_scaler.joblib"
    model_path = MODELS_DIR / "mouse_svm.joblib"
    meta_path = MODELS_DIR / "mouse_meta.json"

    dump(scaler, scaler_path)
    dump(svm_final, model_path)

    meta = {
        "classes": sorted(list(df["user_id"].unique())),
        "features": FEATURE_COLS,
        "internal_acc": float(best_acc),
        "train_csv": str(TRAIN_CSV),
        "C": best_C,
        "gamma": best_gamma,
        "kernel": "rbf",
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("\nSAVED:")
    print(f"  scaler -> {scaler_path}")
    print(f"  model  -> {model_path}")
    print(f"  meta   -> {meta_path}")


if __name__ == "__main__":
    main()
