

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
ART_DIR = BASE_DIR / "artifacts" / "mouse"

TRAIN_CSV = DATA_DIR / "mouse_windows_train.csv"
TEST_CSV = DATA_DIR / "mouse_windows_test.csv"

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

MODELS: Dict[str, Pipeline] = {
    "logreg_baseline": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
    "svm_rbf_final": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=100.0,
                    gamma=0.3,
                    class_weight="balanced",
                ),
            ),
        ]
    ),
}


def _load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    needed = [LABEL_COL, GROUP_COL] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = df[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(str)
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    return df


def _filter_common_users(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_counts = train_df[LABEL_COL].value_counts()
    test_counts = test_df[LABEL_COL].value_counts()

    users = set(train_counts[train_counts >= MIN_PER_USER].index)
    users &= set(test_counts[test_counts >= 1].index)

    train_df = train_df[train_df[LABEL_COL].isin(users)].copy()
    test_df = test_df[test_df[LABEL_COL].isin(users)].copy()
    return train_df, test_df


def _xyg(df: pd.DataFrame):
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[LABEL_COL].to_numpy()
    g = df[GROUP_COL].to_numpy()
    return X, y, g


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)

    train_df = _load_split(TRAIN_CSV)
    test_df = _load_split(TEST_CSV)
    train_df, test_df = _filter_common_users(train_df, test_df)

    X_dev, y_dev, g_dev = _xyg(train_df)
    X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
    y_test = test_df[LABEL_COL].to_numpy()

    print("[INFO] Mouse baseline / CV evaluation")
    print(f"[INFO] Development rows: {len(train_df)}")
    print(f"[INFO] Held-out test rows: {len(test_df)}")
    print(f"[INFO] Users retained: {train_df[LABEL_COL].nunique()}")
    print(f"[INFO] Unique development files: {train_df[GROUP_COL].nunique()}")

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_rows = []
    summary_rows = []

    for model_name, model in MODELS.items():
        print(f"\n[INFO] Running grouped CV for: {model_name}")
        fold_accs = []
        fold_f1s = []

        for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_dev, y_dev, groups=g_dev), start=1):
            X_tr, X_va = X_dev[tr_idx], X_dev[va_idx]
            y_tr, y_va = y_dev[tr_idx], y_dev[va_idx]

            clf = clone(model)
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_va)

            acc = accuracy_score(y_va, pred)
            macro_f1 = f1_score(y_va, pred, average="macro")

            fold_accs.append(acc)
            fold_f1s.append(macro_f1)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "cv_accuracy": float(acc),
                    "cv_macro_f1": float(macro_f1),
                    "n_train": int(len(tr_idx)),
                    "n_val": int(len(va_idx)),
                }
            )
            print(f"  fold={fold_idx} acc={acc:.4f} macro_f1={macro_f1:.4f}")

        final_clf = clone(model)
        final_clf.fit(X_dev, y_dev)
        test_pred = final_clf.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_macro_f1 = f1_score(y_test, test_pred, average="macro")

        summary_rows.append(
            {
                "model": model_name,
                "cv_mean_accuracy": float(np.mean(fold_accs)),
                "cv_std_accuracy": float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0,
                "cv_mean_macro_f1": float(np.mean(fold_f1s)),
                "cv_std_macro_f1": float(np.std(fold_f1s, ddof=1)) if len(fold_f1s) > 1 else 0.0,
                "heldout_test_accuracy": float(test_acc),
                "heldout_test_macro_f1": float(test_macro_f1),
                "n_dev_rows": int(len(train_df)),
                "n_test_rows": int(len(test_df)),
                "n_users": int(train_df[LABEL_COL].nunique()),
            }
        )

        print(
            f"[INFO] {model_name}: cv_mean_acc={np.mean(fold_accs):.4f}, "
            f"cv_mean_macro_f1={np.mean(fold_f1s):.4f}, "
            f"heldout_test_acc={test_acc:.4f}, heldout_test_macro_f1={test_macro_f1:.4f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)

    folds_csv = ART_DIR / "mouse_cv_folds.csv"
    summary_csv = ART_DIR / "mouse_cv_summary.csv"
    summary_json = ART_DIR / "mouse_cv_summary.json"

    fold_df.to_csv(folds_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print("\n[INFO] Wrote outputs:")
    print(f"  {folds_csv}")
    print(f"  {summary_csv}")
    print(f"  {summary_json}")


if __name__ == "__main__":
    main()