# Apps/behavior/train_mouse_xgb.py
"""
Train a mouse classifier using XGBoost on window-level features.

Inputs (prefers augmented, falls back to original):
  - Data/mouse_windows_train_augmented.csv  (if exists)
  - otherwise Data/mouse_windows_train.csv

Outputs (same filenames as SVM so the runtime code keeps working):
  - Models/mouse/mouse_scaler.joblib
  - Models/mouse/mouse_svm.joblib      (actually an XGBClassifier)
  - Models/mouse/mouse_meta.json

Aligned with Apps/behavior/eval_mouse.py and views._get_mouse_model().
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models" / "mouse"

AUG_CSV = DATA_DIR / "mouse_windows_train_augmented.csv"
RAW_CSV = DATA_DIR / "mouse_windows_train.csv"

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
TEST_FRACTION = 0.20  # internal holdout for sanity check, not the official test


def _load_train_df() -> pd.DataFrame:
    if AUG_CSV.is_file():
        train_csv = AUG_CSV
    else:
        train_csv = RAW_CSV

    print(f"[INFO] Using mouse training CSV: {train_csv}")
    if not train_csv.is_file():
        raise SystemExit(f"Mouse train CSV not found at {train_csv}")

    df = pd.read_csv(train_csv)

    if LABEL_COL not in df.columns:
        raise SystemExit(f"Expected '{LABEL_COL}' column in {train_csv.name}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{train_csv.name} is missing feature columns: {missing}")

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df = df.dropna(subset=FEATURE_COLS).copy()

    vc = df[LABEL_COL].value_counts()
    keep_users = vc[vc >= MIN_PER_USER].index
    df = df[df[LABEL_COL].isin(keep_users)].copy()

    if df.empty:
        raise SystemExit("After MIN_PER_USER filtering, no users remain in mouse data.")

    print(
        f"[INFO] Loaded {len(df)} windows for "
        f"{df[LABEL_COL].nunique()} users from {train_csv}"
    )
    return df, train_csv


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df, train_csv = _load_train_df()
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[LABEL_COL].astype(str).to_numpy()

    # ------------------------------------------------------------------
    # Train/val split for hyperparameter selection
    # ------------------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_FRACTION,
        random_state=42,
        stratify=y,
    )

    # Scale features (not strictly needed for trees, but keeps runtime
    # pipeline identical: scaler + model).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # ------------------------------------------------------------------
    # Small hyperparameter grid for XGBoost (fast but non-trivial)
    # ------------------------------------------------------------------
    param_grid = []
    for lr in [0.05, 0.1]:
        for depth in [3, 5]:
            for n_estimators in [100, 200]:
                param_grid.append(
                    {"learning_rate": lr, "max_depth": depth, "n_estimators": n_estimators}
                )

    best_acc = -1.0
    best_params = None

    num_classes = len(np.unique(y))
    print(f"[INFO] Tuning XGBoost on internal holdout (classes={num_classes})...")

    for params in param_grid:
        print(
            f"  trying lr={params['learning_rate']}, "
            f"depth={params['max_depth']}, "
            f"n_estimators={params['n_estimators']}..."
        )

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_val_pred)
        print(f"    -> val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_params = params

    if best_params is None:
        raise SystemExit("Hyperparameter search failed; no best params found.")

    print(
        "\n[INFO] Best XGB params:"
        f"\n       learning_rate = {best_params['learning_rate']}"
        f"\n       max_depth     = {best_params['max_depth']}"
        f"\n       n_estimators  = {best_params['n_estimators']}"
        f"\n       internal_acc  = {best_acc:.4f}"
    )

    # ------------------------------------------------------------------
    # Retrain on all data with best params
    # ------------------------------------------------------------------
    print("[INFO] Training final XGBoost model on full dataset...")
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X)

    final_model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    final_model.fit(X_full_scaled, y)

    print("[INFO] Final XGBoost training done.")

    # ------------------------------------------------------------------
    # Optional: classification report on the val set using final model
    # (sanity check only; doesn't affect saved model)
    # ------------------------------------------------------------------
    y_val_pred_final = final_model.predict(scaler_full.transform(X_val))
    print("\n[INFO] Classification report on internal holdout (final model):")
    print(classification_report(y_val, y_val_pred_final))

    # ------------------------------------------------------------------
    # Save artifacts – NOTE: filenames unchanged so runtime keeps working
    # ------------------------------------------------------------------
    scaler_path = MODELS_DIR / "mouse_scaler.joblib"
    model_path = MODELS_DIR / "mouse_svm.joblib"  # actually XGB, name kept for compat
    meta_path = MODELS_DIR / "mouse_meta.json"

    dump(scaler_full, scaler_path)
    dump(final_model, model_path)

    meta = {
        "classes": sorted(list(df[LABEL_COL].unique())),
        "features": FEATURE_COLS,
        "internal_acc": float(best_acc),
        "train_csv": str(train_csv),
        "model_type": "xgboost",
        "learning_rate": best_params["learning_rate"],
        "max_depth": best_params["max_depth"],
        "n_estimators": best_params["n_estimators"],
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("\nSAVED:")
    print(f"  scaler -> {scaler_path}")
    print(f"  model  -> {model_path} (XGBClassifier)")
    print(f"  meta   -> {meta_path}")
    print(f"[INFO] internal_acc (val) = {best_acc:.4f}")


if __name__ == "__main__":
    main()
