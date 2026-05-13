# Apps/behavior/train_mouse_svm.py
"""
Train the final mouse SVM classifier.

Clean training story:
  1) Tune hyperparameters on the base training set only
     -> Data/mouse_windows_train.csv
  2) Train the final runtime model on the augmented full training set
     -> Data/mouse_windows_train_augmented.csv
  3) Evaluate later on the separate held-out test set using eval_mouse.py
     -> Data/mouse_windows_test.csv

Outputs:
  - Models/mouse/mouse_scaler.joblib
  - Models/mouse/mouse_svm.joblib
  - Models/mouse/mouse_meta.json
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models" / "mouse"

TUNE_CSV = DATA_DIR / "mouse_windows_train.csv"
FINAL_TRAIN_CSV = DATA_DIR / "mouse_windows_train_augmented.csv"

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

LABEL_COL = "user_id"
MIN_PER_USER = 40
TEST_FRACTION = 0.25  # internal holdout only for tuning / sanity check

# Narrowed hyperparameter search based on previous run history.
# Earlier wider search already explored smaller C values (1, 5, 10, 20)
# and showed the best result near the upper edge around C=50, gamma=0.2.
# To save time, this run searches only around that stronger region.
C_VALUES = [50.0, 80.0, 100.0]
GAMMAS = [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_df(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Mouse train CSV not found: {path}")

    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        raise SystemExit(f"Expected '{LABEL_COL}' column in {path.name}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing feature columns: {missing}")

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLS).copy()

    vc = df[LABEL_COL].value_counts()
    keep_users = vc[vc >= MIN_PER_USER].index
    df = df[df[LABEL_COL].isin(keep_users)].copy()

    if df.empty:
        raise SystemExit(f"After MIN_PER_USER filtering, no users remain in {path.name}.")

    return df


def _restrict_to_common_users(tune_df: pd.DataFrame, final_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tune_users = set(tune_df[LABEL_COL].unique())
    final_users = set(final_df[LABEL_COL].unique())
    common = sorted(tune_users & final_users)
    if not common:
        raise SystemExit("No common users between mouse_windows_train.csv and mouse_windows_train_augmented.csv")

    tune_df = tune_df[tune_df[LABEL_COL].isin(common)].copy()
    final_df = final_df[final_df[LABEL_COL].isin(common)].copy()
    return tune_df, final_df


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    tune_df = _load_df(TUNE_CSV)
    final_df = _load_df(FINAL_TRAIN_CSV)
    tune_df, final_df = _restrict_to_common_users(tune_df, final_df)

    print(
        f"[INFO] Loaded {len(tune_df)} tuning windows for "
        f"{tune_df[LABEL_COL].nunique()} users from {TUNE_CSV}"
    )
    print(
        f"[INFO] Loaded {len(final_df)} final-train windows for "
        f"{final_df[LABEL_COL].nunique()} users from {FINAL_TRAIN_CSV}"
    )

    # -----------------------------------------------------------------
    # Hyperparameter tuning on the BASE training set only
    # -----------------------------------------------------------------
    X_tune = tune_df[FEATURE_COLS].to_numpy(dtype=float)
    y_tune = tune_df[LABEL_COL].astype(str).to_numpy()

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_tune,
        y_tune,
        test_size=TEST_FRACTION,
        random_state=42,
        stratify=y_tune,
    )

    scaler_tune = StandardScaler()
    X_train_scaled = scaler_tune.fit_transform(X_train)
    X_holdout_scaled = scaler_tune.transform(X_holdout)

    best_C = None
    best_gamma = None
    best_acc = -1.0

    print("[INFO] Tuning SVM hyperparameters on base-train internal holdout...")
    for C in C_VALUES:
        for gamma in GAMMAS:
            svm = SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                decision_function_shape="ovr",
                class_weight="balanced",
                probability=False,
                cache_size=1000,
            )
            svm.fit(X_train_scaled, y_train)
            acc = svm.score(X_holdout_scaled, y_holdout)
            print(f"  C={C:<5} gamma={str(gamma):<6} -> acc={acc:.4f}")
            if acc > best_acc:
                best_acc = float(acc)
                best_C = C
                best_gamma = gamma

    print(
        f"[INFO] Best hyperparams: C={best_C}, gamma={best_gamma}, "
        f"internal_acc={best_acc:.4f}"
    )

    # -----------------------------------------------------------------
    # Final runtime model training on the AUGMENTED full training set
    # -----------------------------------------------------------------
    X_final = final_df[FEATURE_COLS].to_numpy(dtype=float)
    y_final = final_df[LABEL_COL].astype(str).to_numpy()

    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    svm_final = SVC(
        kernel="rbf",
        C=best_C,
        gamma=best_gamma,
        decision_function_shape="ovr",
        class_weight="balanced",
        probability=True,  # needed at runtime for mouse trust
        cache_size=1000,
    )

    print("[INFO] Training final mouse SVM on augmented full train set...")
    svm_final.fit(X_final_scaled, y_final)
    print("[INFO] Final mouse SVM training done.")

    # -----------------------------------------------------------------
    # Optional sanity report on the BASE holdout using the FINAL model
    # -----------------------------------------------------------------
    common_holdout_mask = np.isin(y_holdout, svm_final.classes_)
    X_holdout_common = X_holdout[common_holdout_mask]
    y_holdout_common = y_holdout[common_holdout_mask]

    if len(X_holdout_common):
        y_holdout_pred = svm_final.predict(scaler_final.transform(X_holdout_common))
        print("\n[INFO] Classification report on base-train internal holdout (sanity only):")
        print(classification_report(y_holdout_common, y_holdout_pred))
        sanity_acc = accuracy_score(y_holdout_common, y_holdout_pred)
    else:
        sanity_acc = None
        print("\n[INFO] No compatible holdout rows available for sanity classification report.")

    # -----------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------
    scaler_path = MODELS_DIR / "mouse_scaler.joblib"
    model_path = MODELS_DIR / "mouse_svm.joblib"
    meta_path = MODELS_DIR / "mouse_meta.json"

    dump(scaler_final, scaler_path)
    dump(svm_final, model_path)

    meta = {
        "classes": sorted(list(map(str, svm_final.classes_))),
        "features": FEATURE_COLS,
        "tune_csv": str(TUNE_CSV),
        "final_train_csv": str(FINAL_TRAIN_CSV),
        "train_csv": str(FINAL_TRAIN_CSV),
        "internal_holdout_acc": float(best_acc),
        "internal_acc": float(best_acc),
        "sanity_holdout_acc": None if sanity_acc is None else float(sanity_acc),
        "C": best_C,
        "gamma": best_gamma,
        "kernel": "rbf",
        "min_per_user": MIN_PER_USER,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("\nSAVED:")
    print(f"  scaler -> {scaler_path}")
    print(f"  model  -> {model_path}")
    print(f"  meta   -> {meta_path}")


if __name__ == "__main__":
    main()
