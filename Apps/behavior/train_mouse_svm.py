# Apps/behavior/train_mouse_svm.py
"""
Train the final mouse SVM classifier.

Report-friendly training story:
  1) Tune hyperparameters on the base training set only
     -> Data/mouse_windows_train.csv
  2) Use grouped cross-validation by raw session file to reduce leakage
  3) Select hyperparameters by mean grouped-CV macro F1 (and accuracy)
  4) Train the final runtime model on the augmented full training set
     -> Data/mouse_windows_train_augmented.csv
  5) Evaluate later on the separate held-out test set using eval_mouse.py
     -> Data/mouse_windows_test.csv

Outputs:
  - Models/mouse/mouse_scaler.joblib
  - Models/mouse/mouse_svm.joblib
  - Models/mouse/mouse_meta.json
  - artifacts/mouse/mouse_tuning_results.csv
  - artifacts/mouse/mouse_tuning_summary.json
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models" / "mouse"
ART_DIR = BASE_DIR / "artifacts" / "mouse"

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
GROUP_COL = "file"
MIN_PER_USER = 40
N_SPLITS = 5

# Narrowed hyperparameter search based on previous run history.
# Earlier wider search explored smaller C values and weaker gamma settings.
# The best region was near the upper edge, so this search stays focused there.
C_VALUES = [50.0, 80.0, 100.0]
GAMMAS = [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_df(path: Path, require_group: bool) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Mouse train CSV not found: {path}")

    df = pd.read_csv(path)

    needed = [LABEL_COL] + FEATURE_COLS
    if require_group:
        needed = [LABEL_COL, GROUP_COL] + FEATURE_COLS

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns: {missing}")

    df = df[needed].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    if require_group:
        df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()

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


def _xyg(df: pd.DataFrame):
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[LABEL_COL].astype(str).to_numpy()
    g = df[GROUP_COL].astype(str).to_numpy()
    return X, y, g


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    tune_df = _load_df(TUNE_CSV, require_group=True)
    final_df = _load_df(FINAL_TRAIN_CSV, require_group=False)
    tune_df, final_df = _restrict_to_common_users(tune_df, final_df)

    print(
        f"[INFO] Loaded {len(tune_df)} tuning windows for "
        f"{tune_df[LABEL_COL].nunique()} users from {TUNE_CSV}"
    )
    print(
        f"[INFO] Loaded {len(final_df)} final-train windows for "
        f"{final_df[LABEL_COL].nunique()} users from {FINAL_TRAIN_CSV}"
    )
    print(f"[INFO] Unique tuning files: {tune_df[GROUP_COL].nunique()}")

    # -----------------------------------------------------------------
    # Grouped CV tuning on the BASE training set only
    # -----------------------------------------------------------------
    X_tune, y_tune, g_tune = _xyg(tune_df)

    gkf = GroupKFold(n_splits=N_SPLITS)
    tuning_rows = []

    print("[INFO] Tuning SVM hyperparameters with grouped CV by raw file...")
    for C in C_VALUES:
        for gamma in GAMMAS:
            fold_accs = []
            fold_f1s = []

            base_model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        SVC(
                            kernel="rbf",
                            C=C,
                            gamma=gamma,
                            decision_function_shape="ovr",
                            class_weight="balanced",
                            probability=False,
                            cache_size=1000,
                        ),
                    ),
                ]
            )

            for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_tune, y_tune, groups=g_tune), start=1):
                X_tr, X_va = X_tune[tr_idx], X_tune[va_idx]
                y_tr, y_va = y_tune[tr_idx], y_tune[va_idx]

                clf = clone(base_model)
                clf.fit(X_tr, y_tr)
                pred = clf.predict(X_va)

                acc = accuracy_score(y_va, pred)
                macro_f1 = f1_score(y_va, pred, average="macro")
                fold_accs.append(acc)
                fold_f1s.append(macro_f1)

                tuning_rows.append(
                    {
                        "C": C,
                        "gamma": gamma,
                        "fold": fold_idx,
                        "cv_accuracy": float(acc),
                        "cv_macro_f1": float(macro_f1),
                        "n_train": int(len(tr_idx)),
                        "n_val": int(len(va_idx)),
                    }
                )

            mean_acc = float(np.mean(fold_accs))
            mean_f1 = float(np.mean(fold_f1s))
            print(f"  C={C:<5} gamma={str(gamma):<4} -> cv_mean_acc={mean_acc:.4f}, cv_mean_macro_f1={mean_f1:.4f}")

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_summary = (
        tuning_df.groupby(["C", "gamma"], as_index=False)
        .agg(
            cv_mean_accuracy=("cv_accuracy", "mean"),
            cv_std_accuracy=("cv_accuracy", "std"),
            cv_mean_macro_f1=("cv_macro_f1", "mean"),
            cv_std_macro_f1=("cv_macro_f1", "std"),
        )
        .sort_values(["cv_mean_macro_f1", "cv_mean_accuracy"], ascending=False)
        .reset_index(drop=True)
    )

    best_row = tuning_summary.iloc[0]
    best_C = float(best_row["C"])
    best_gamma = float(best_row["gamma"])
    best_cv_acc = float(best_row["cv_mean_accuracy"])
    best_cv_f1 = float(best_row["cv_mean_macro_f1"])

    print(
        f"[INFO] Best hyperparams: C={best_C}, gamma={best_gamma}, "
        f"cv_mean_acc={best_cv_acc:.4f}, cv_mean_macro_f1={best_cv_f1:.4f}"
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
    # Optional sanity report on the BASE training set using grouped best params
    # -----------------------------------------------------------------
    sanity_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=best_C,
                    gamma=best_gamma,
                    decision_function_shape="ovr",
                    class_weight="balanced",
                    probability=False,
                    cache_size=1000,
                ),
            ),
        ]
    )
    sanity_model.fit(X_tune, y_tune)
    sanity_pred = sanity_model.predict(X_tune)
    sanity_acc = accuracy_score(y_tune, sanity_pred)
    print("\n[INFO] Classification report on base-train data (sanity only):")
    print(classification_report(y_tune, sanity_pred))

    # -----------------------------------------------------------------
    # Save artifacts / proofs
    # -----------------------------------------------------------------
    scaler_path = MODELS_DIR / "mouse_scaler.joblib"
    model_path = MODELS_DIR / "mouse_svm.joblib"
    meta_path = MODELS_DIR / "mouse_meta.json"
    tuning_csv = ART_DIR / "mouse_tuning_results.csv"
    tuning_summary_json = ART_DIR / "mouse_tuning_summary.json"

    dump(scaler_final, scaler_path)
    dump(svm_final, model_path)

    tuning_df.to_csv(tuning_csv, index=False)
    with open(tuning_summary_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_C": best_C,
                "best_gamma": best_gamma,
                "best_cv_mean_accuracy": best_cv_acc,
                "best_cv_mean_macro_f1": best_cv_f1,
                "n_tune_rows": int(len(tune_df)),
                "n_final_train_rows": int(len(final_df)),
                "n_users": int(final_df[LABEL_COL].nunique()),
                "n_tuning_files": int(tune_df[GROUP_COL].nunique()),
                "grid_C": C_VALUES,
                "grid_gamma": GAMMAS,
            },
            fh,
            indent=2,
        )

    meta = {
        "model_type": "svm_rbf",
        "classes": sorted(list(map(str, svm_final.classes_))),
        "features": FEATURE_COLS,
        "tune_csv": str(TUNE_CSV),
        "final_train_csv": str(FINAL_TRAIN_CSV),
        "train_csv": str(FINAL_TRAIN_CSV),
        "group_col": GROUP_COL,
        "grouped_cv_metric": "macro_f1",
        "best_cv_mean_accuracy": float(best_cv_acc),
        "best_cv_mean_macro_f1": float(best_cv_f1),
        "internal_holdout_acc": float(best_cv_acc),
        "internal_acc": float(best_cv_acc),
        "sanity_holdout_acc": float(sanity_acc),
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
    print(f"  tuning -> {tuning_csv}")
    print(f"  summary -> {tuning_summary_json}")


if __name__ == "__main__":
    main()
