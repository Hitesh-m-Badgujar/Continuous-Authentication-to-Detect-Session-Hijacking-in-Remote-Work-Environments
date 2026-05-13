from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models" / "kb_svm"
ART_DIR = BASE_DIR / "artifacts"

TRAIN_CSV = DATA_DIR / "keyboard_train_windows.csv"
VAL_CSV = DATA_DIR / "keyboard_val_windows.csv"
TEST_CSV = DATA_DIR / "keyboard_test_windows.csv"

FEATURE_COLS = [
    "dwell_mean", "dwell_std", "dwell_p10", "dwell_p50", "dwell_p90",
    "dd_mean", "dd_std", "dd_p10", "dd_p50", "dd_p90",
    "ud_mean", "ud_std", "ud_p10", "ud_p50", "ud_p90",
    "backspace_rate", "burst_mean", "idle_frac",
]

LABEL_COL = "user_id"
MIN_PER_USER = 50

GRID = [
    (1.0, "scale"),
    (5.0, "scale"),
    (10.0, "scale"),
    (20.0, "scale"),
    (10.0, 0.1),
    (20.0, 0.1),
]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    needed = [LABEL_COL] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = df[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _filter_common_users(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_counts = train_df[LABEL_COL].value_counts()
    val_counts = val_df[LABEL_COL].value_counts()
    test_counts = test_df[LABEL_COL].value_counts()

    users = set(train_counts[train_counts >= MIN_PER_USER].index)
    users &= set(val_counts[val_counts >= 1].index)
    users &= set(test_counts[test_counts >= 1].index)

    train_df = train_df[train_df[LABEL_COL].isin(users)].copy()
    val_df = val_df[val_df[LABEL_COL].isin(users)].copy()
    test_df = test_df[test_df[LABEL_COL].isin(users)].copy()

    return train_df, val_df, test_df


def _xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[LABEL_COL].astype(str).to_numpy()
    return X, y


def _per_user_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(pd.unique(y_true))
    rows = []

    for lab in labels:
        genuine = (y_true == lab)
        pred_genuine = (y_pred == lab)

        tp = int(np.sum(genuine & pred_genuine))
        fn = int(np.sum(genuine & ~pred_genuine))
        fp = int(np.sum(~genuine & pred_genuine))
        tn = int(np.sum(~genuine & ~pred_genuine))

        far = fp / (fp + tn) if (fp + tn) else 0.0
        frr = fn / (fn + tp) if (fn + tp) else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

        rows.append({
            "user_id": lab,
            "FAR": float(far),
            "FRR": float(frr),
            "ACC": float(acc),
            "n_genuine": int(tp + fn),
            "n_impostor": int(fp + tn),
        })

    return pd.DataFrame(rows)


def main() -> None:
    print("[INFO] eval_kb_svm (explicit split) starting")
    print(f"[INFO] Train file: {TRAIN_CSV}")
    print(f"[INFO] Val file:   {VAL_CSV}")
    print(f"[INFO] Test file:  {TEST_CSV}")

    train_df = _load_csv(TRAIN_CSV)
    val_df = _load_csv(VAL_CSV)
    test_df = _load_csv(TEST_CSV)

    train_df, val_df, test_df = _filter_common_users(train_df, val_df, test_df)

    users = sorted(train_df[LABEL_COL].unique())
    print(f"[INFO] Users retained: {len(users)}")
    print(f"[INFO] Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    print("[INFO] Tuning SVM hyperparameters...")
    best_acc = -1.0
    best_params = None

    for C, gamma in GRID:
        clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")
        clf.fit(X_train_s, y_train)
        pred_val = clf.predict(X_val_s)
        val_acc = accuracy_score(y_val, pred_val)
        print(f"  C={C:<5} gamma={str(gamma):<6} -> val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = float(val_acc)
            best_params = (C, gamma)

    assert best_params is not None
    best_C, best_gamma = best_params
    print(f"[INFO] Best hyperparams: C={best_C}, gamma={best_gamma}, val_acc={best_acc:.4f}")

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    final_scaler = StandardScaler()
    X_trainval_s = final_scaler.fit_transform(X_trainval)
    X_test_s = final_scaler.transform(X_test)

    print("[INFO] Training final SVM with best hyperparams on train+val...")
    final_clf = SVC(kernel="rbf", C=best_C, gamma=best_gamma, class_weight="balanced")
    final_clf.fit(X_trainval_s, y_trainval)
    print("[INFO] Final SVM training done")

    y_pred = final_clf.predict(X_test_s)
    metrics_df = _per_user_metrics(y_test, y_pred)

    for _, row in metrics_df.iterrows():
        print(
            f"  {row['user_id']}  FAR={row['FAR']:.3f}  FRR={row['FRR']:.3f}  "
            f"ACC={row['ACC']:.3f}  n_g={int(row['n_genuine'])} n_i={int(row['n_impostor'])}"
        )

    macro_far = float(metrics_df["FAR"].mean()) if len(metrics_df) else 0.0
    macro_frr = float(metrics_df["FRR"].mean()) if len(metrics_df) else 0.0
    macro_acc = float(metrics_df["ACC"].mean()) if len(metrics_df) else 0.0

    print(f"\nMACRO FAR={macro_far:.3f} FRR={macro_frr:.3f} ACC={macro_acc:.3f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    metrics_csv = ART_DIR / "kb_svm_eval_metrics.csv"
    overview_json = ART_DIR / "kb_svm_eval_overview.json"
    model_path = MODELS_DIR / "kb_svm_model.joblib"
    scaler_path = MODELS_DIR / "kb_svm_scaler.joblib"

    metrics_df.to_csv(metrics_csv, index=False)

    with open(overview_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "macro_far": macro_far,
                "macro_frr": macro_frr,
                "macro_acc": macro_acc,
                "best_C": best_C,
                "best_gamma": best_gamma,
                "n_users": len(users),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "feature_cols": FEATURE_COLS,
            },
            f,
            indent=2,
        )

    joblib.dump(final_clf, model_path)
    joblib.dump(final_scaler, scaler_path)

    print(f"\nWROTE SVM metrics CSV -> {metrics_csv}")
    print(f"WROTE SVM macro JSON  -> {overview_json}")
    print(f"WROTE SVM model + scaler -> {MODELS_DIR}")


if __name__ == "__main__":
    main()