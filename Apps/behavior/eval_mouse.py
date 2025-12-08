# Apps/behavior/eval_mouse.py
"""
Offline evaluation for the mouse SVM model.

- Loads SVM + scaler from Models/mouse
- Uses Data/mouse_windows_test.csv (window-level features)
- For each user:
    * treats that user's windows as genuine
    * treats all other users' windows as impostors
    * computes FAR / FRR / ACC
- Writes:
    - artifacts/mouse_svm_eval_metrics.csv
    - artifacts/mouse_svm_eval_overview.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load

# ---------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models" / "mouse"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = DATA_DIR / "mouse_windows_test.csv"
SCALER_PATH = MODELS_DIR / "mouse_scaler.joblib"
MODEL_PATH = MODELS_DIR / "mouse_svm.joblib"
META_PATH = MODELS_DIR / "mouse_meta.json"

# These must match your mouse feature CSV columns
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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_model_and_scaler():
    """Load scaler + SVM model, unwrapping dicts if necessary."""
    if not SCALER_PATH.is_file():
        raise SystemExit(f"Missing scaler at {SCALER_PATH}")
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Missing model at {MODEL_PATH}")

    scaler = joblib_load(SCALER_PATH)
    model_obj = joblib_load(MODEL_PATH)

    # Handle both "plain SVC" and "wrapped in dict"
    if isinstance(model_obj, dict):
        if "model" in model_obj:
            model = model_obj["model"]
        elif "svc" in model_obj:
            model = model_obj["svc"]
        else:
            raise SystemExit(
                f"Loaded model from {MODEL_PATH} is a dict without 'model' or 'svc' key."
            )
    else:
        model = model_obj

    # Sanity check
    if not hasattr(model, "predict"):
        raise SystemExit(
            f"Loaded object from {MODEL_PATH} does not have .predict; type={type(model)}"
        )

    return scaler, model


def _compute_far_frr_acc_for_user(
    uid: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, int, int]:
    """
    Compute FAR/FRR/ACC from multi-class predictions by treating
    'uid' as the genuine class and all others as impostors.

    FAR = P(impostor accepted as uid)
    FRR = P(genuine uid rejected)
    ACC = 1 - 0.5 * (FAR + FRR)
    """
    is_genuine = (y_true == uid)
    is_impostor = ~is_genuine

    n_g = int(is_genuine.sum())
    n_i = int(is_impostor.sum())

    if n_g == 0 or n_i == 0:
        return np.nan, np.nan, np.nan, n_g, n_i

    pred_uid = (y_pred == uid)

    FAR = float(np.mean(pred_uid[is_impostor]))       # impostor accepted
    FRR = float(np.mean(~pred_uid[is_genuine]))       # genuine rejected
    ACC = 1.0 - 0.5 * (FAR + FRR)

    return FAR, FRR, ACC, n_g, n_i


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("[INFO] eval_mouse (SVM) starting")
    print(f"[INFO] Loading test data from {DATA_CSV}")

    if not DATA_CSV.is_file():
        raise SystemExit(f"Mouse test CSV not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)

    if "user_id" not in df.columns:
        raise SystemExit("Expected 'user_id' column in mouse_windows_test.csv")

    df["user_id"] = df["user_id"].astype(str).str.strip()
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Mouse test CSV is missing feature columns: {missing}")

    df = df.dropna(subset=FEATURE_COLS).copy()

    users = sorted(df["user_id"].unique())
    print(f"[INFO] Evaluating {len(users)} users")
    if not users:
        raise SystemExit("No users found in mouse_windows_test.csv")

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["user_id"].astype(str).to_numpy()

    scaler, model = _load_model_and_scaler()
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    rows = []
    for uid in users:
        FAR, FRR, ACC, n_g, n_i = _compute_far_frr_acc_for_user(uid, y, y_pred)
        if np.isnan(ACC):
            print(f"[WARN] Skipping user {uid}: insufficient data (n_g={n_g}, n_i={n_i})")
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

    metrics_csv = ARTIFACTS_DIR / "mouse_svm_eval_metrics.csv"
    overview_json = ARTIFACTS_DIR / "mouse_svm_eval_overview.json"

    df_rows = pd.DataFrame(
        rows,
        columns=["user_id", "FAR", "FRR", "ACC", "n_genuine", "n_impostor"],
    )
    df_rows.to_csv(metrics_csv, index=False)

    overview = {
        "macro_FAR": macro_FAR,
        "macro_FRR": macro_FRR,
        "macro_ACC": macro_ACC,
        "n_users": len(rows),
        "data_csv": str(DATA_CSV),
        "features": FEATURE_COLS,
        "model": "mouse SVM (multi-class, evaluated per-user genuine vs impostor)",
    }

    with open(overview_json, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    print(f"\nWROTE mouse SVM metrics CSV -> {metrics_csv}")
    print(f"WROTE mouse SVM macro JSON  -> {overview_json}")


if __name__ == "__main__":
    main()
