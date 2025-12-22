from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# IMPORTANT: use CMU-derived windows
DATA_CSV = Path("Data/kb_cmu_windows.csv")
MODELS_DIR = Path("Models/kb_svm")
ARTIFACTS_DIR = Path("artifacts")

FEATURES: List[str] = [
    "dwell_mean",
    "dwell_std",
    "dwell_p10",
    "dwell_p50",
    "dwell_p90",
    "dd_mean",
    "dd_std",
    "dd_p10",
    "dd_p50",
    "dd_p90",
    "ud_mean",
    "ud_std",
    "ud_p10",
    "ud_p50",
    "ud_p90",
    "backspace_rate",
    "burst_mean",
    "idle_frac",
]

MIN_PER_USER = 40
TRAIN_FRACTION = 0.7  # more train data than before


def _load_data() -> pd.DataFrame:
    print(f"[INFO] Loading data from {DATA_CSV}")
    if not DATA_CSV.exists():
        raise SystemExit(f"Data file not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    if df.empty:
        raise SystemExit(f"{DATA_CSV.name} exists but is empty.")

    if "user_id" not in df.columns:
        raise SystemExit("Expected 'user_id' column in kb_cmu_windows.csv.")

    df["user_id"] = df["user_id"].astype(str).str.strip()
    df = df.dropna(subset=FEATURES).copy()

    vc = df["user_id"].value_counts()
    keep = vc[vc >= MIN_PER_USER].index
    df = df[df["user_id"].isin(keep)].copy()

    if df.empty:
        print("User counts BEFORE filter:")
        print(vc.sort_index().to_string())
        raise SystemExit("After MIN_PER_USER filter no users remain.")

    sort_cols = [c for c in ["session_id", "start_idx"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    else:
        df = df.sort_index()

    print(f"[INFO] Loaded {len(df)} rows for {df['user_id'].nunique()} users")
    return df


def _split_train_test(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-user chronological split:
      - First TRAIN_FRACTION windows -> train
      - Remaining -> test
    """
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    users = sorted(df["user_id"].unique())
    for uid in users:
        u_df = df[df["user_id"] == uid]
        n_total = len(u_df)
        n_train = max(1, int(n_total * TRAIN_FRACTION))
        if n_total - n_train < 1:
            continue

        train_df = u_df.iloc[:n_train]
        test_df = u_df.iloc[n_train:]

        X_train_list.append(train_df[FEATURES].to_numpy(dtype=float))
        y_train_list.append(train_df["user_id"].to_numpy())

        X_test_list.append(test_df[FEATURES].to_numpy(dtype=float))
        y_test_list.append(test_df["user_id"].to_numpy())

    if not X_train_list:
        raise SystemExit("Train/test split produced no data.")

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    return X_train, y_train, X_test, y_test


def _compute_far_frr_acc_for_user(
    uid: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, int, int]:
    """
    Compute FAR/FRR/ACC from multi-class predictions by treating
    'uid' as the genuine class and all others as impostors.
    """
    is_genuine = (y_test == uid)
    is_impostor = ~is_genuine

    if not np.any(is_genuine) or not np.any(is_impostor):
        return np.nan, np.nan, np.nan, int(is_genuine.sum()), int(is_impostor.sum())

    pred_uid = (y_pred == uid)

    FAR = float(np.mean(pred_uid[is_impostor]))          # impostor accepted
    FRR = float(np.mean(~pred_uid[is_genuine]))          # genuine rejected
    ACC = 1.0 - 0.5 * (FAR + FRR)
    return FAR, FRR, ACC, int(is_genuine.sum()), int(is_impostor.sum())


def main() -> None:
    print("[INFO] eval_kb_svm (tuned) starting")

    df = _load_data()
    X_train, y_train, X_test, y_test = _split_train_test(df)
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------- Hyperparameter search on a validation split ----------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    param_grid = [
        (1.0, "scale"),
        (5.0, "scale"),
        (10.0, "scale"),
        (20.0, "scale"),
        (10.0, 0.1),
        (20.0, 0.1),
    ]

    best_C = None
    best_gamma = None
    best_val_acc = -1.0

    print("[INFO] Tuning SVM hyperparameters...")
    for C, gamma in param_grid:
        svm = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            decision_function_shape="ovr",
            class_weight="balanced",
        )
        svm.fit(X_tr, y_tr)
        val_acc = svm.score(X_val, y_val)
        print(f"  C={C:<5} gamma={gamma:<6} -> val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_C = C
            best_gamma = gamma

    print(
        f"[INFO] Best hyperparams: C={best_C}, gamma={best_gamma}, "
        f"val_acc={best_val_acc:.4f}"
    )

    # ---------------- Train final SVM on full training data ----------------
    svm = SVC(
        kernel="rbf",
        C=best_C,
        gamma=best_gamma,
        decision_function_shape="ovr",
        class_weight="balanced",
    )
    print("[INFO] Training final SVM with best hyperparams...")
    svm.fit(X_train_scaled, y_train)
    print("[INFO] Final SVM training done")

    # ---------------- Evaluate on test split ----------------
    y_pred = svm.predict(X_test_scaled)

    users = sorted(df["user_id"].unique())
    rows = []

    for uid in users:
        FAR, FRR, ACC, n_g, n_i = _compute_far_frr_acc_for_user(uid, y_test, y_pred)
        if np.isnan(ACC):
            continue
        rows.append((uid, FAR, FRR, ACC, n_g, n_i))
        print(
            f"  {uid:>4s}  FAR={FAR:.3f}  FRR={FRR:.3f}  ACC={ACC:.3f}  "
            f"n_g={n_g} n_i={n_i}"
        )

    if not rows:
        print("[WARN] No users evaluated.")
        return

    FARs = np.array([r[1] for r in rows], dtype=float)
    FRRs = np.array([r[2] for r in rows], dtype=float)
    ACCs = np.array([r[3] for r in rows], dtype=float)

    macro_FAR = float(FARs.mean())
    macro_FRR = float(FRRs.mean())
    macro_ACC = float(ACCs.mean())

    print()
    print(f"MACRO FAR={macro_FAR:.3f} FRR={macro_FRR:.3f} ACC={macro_ACC:.3f}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_csv = ARTIFACTS_DIR / "kb_svm_eval_metrics.csv"
    overview_json = ARTIFACTS_DIR / "kb_svm_eval_overview.json"

    df_rows = pd.DataFrame(
        rows,
        columns=["user_id", "FAR", "FRR", "ACC", "n_genuine", "n_impostor"],
    )
    df_rows.to_csv(metrics_csv, index=False)

    import json

    overview = {
        "macro_FAR": macro_FAR,
        "macro_FRR": macro_FRR,
        "macro_ACC": macro_ACC,
        "n_users": len(rows),
        "data_csv": str(DATA_CSV),
        "features": FEATURES,
        "model": "SVM (RBF, tuned; class_weight=balanced)",
        "train_fraction": TRAIN_FRACTION,
        "best_C": best_C,
        "best_gamma": best_gamma,
        "val_acc_best": best_val_acc,
    }

    with open(overview_json, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    print(f"\nWROTE SVM metrics CSV -> {metrics_csv}")
    print(f"WROTE SVM macro JSON  -> {overview_json}")

    # Save final model + scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(scaler, MODELS_DIR / "kb_svm_scaler.joblib")
    dump(svm, MODELS_DIR / "kb_svm_model.joblib")
    print(f"WROTE SVM model + scaler -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
