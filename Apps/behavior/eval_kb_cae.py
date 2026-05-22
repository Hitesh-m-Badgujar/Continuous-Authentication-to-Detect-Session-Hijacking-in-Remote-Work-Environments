# Apps/behavior/eval_kb_cae.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from tensorflow import keras

# ---------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models" / "kb_cae"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VAL_CSV = DATA_DIR / "keyboard_val_windows.csv"
DEFAULT_TEST_CSV = DATA_DIR / "keyboard_test_windows.csv"
DEFAULT_MODEL_DIR = MODELS_DIR

LABEL_COL = "user_id"
FEATURE_COLS: List[str] = [
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


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------


def _load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing keyboard CAE split: {path}")

    df = pd.read_csv(path)
    needed = [LABEL_COL] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = df[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    return df



def _load_user_model_and_scaler(user_dir: Path) -> Tuple[object, keras.Model]:
    scaler_path = user_dir / "scaler.joblib"
    model_path = user_dir / "cae.keras"

    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing scaler.joblib at {scaler_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing cae.keras at {model_path}")

    scaler = joblib_load(scaler_path)
    model = keras.models.load_model(model_path)
    return scaler, model



def compute_residuals(X: np.ndarray, scaler, model: keras.Model) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled, batch_size=512, verbose=0)
    pred = np.asarray(pred, dtype=np.float32)

    if pred.shape != X_scaled.shape:
        raise RuntimeError(f"Prediction shape {pred.shape} != input shape {X_scaled.shape}")

    err = np.mean((X_scaled - pred) ** 2, axis=1)
    return err


# ---------------------------------------------------------------------
# Metrics / threshold search
# ---------------------------------------------------------------------


def metrics_at_tau(res_g: np.ndarray, res_i: np.ndarray, tau: float) -> Dict[str, float]:
    res_g = np.asarray(res_g)
    res_i = np.asarray(res_i)

    # Accept if residual <= tau
    tp = np.sum(res_g <= tau)
    fn = np.sum(res_g > tau)
    tn = np.sum(res_i > tau)
    fp = np.sum(res_i <= tau)

    n_g = len(res_g)
    n_i = len(res_i)

    far = fp / max(1, n_i)
    frr = fn / max(1, n_g)
    acc = (tp + tn) / max(1, (n_g + n_i))

    return {"FAR": float(far), "FRR": float(frr), "ACC": float(acc)}



def find_best_tau(res_g: np.ndarray, res_i: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    Pick tau on the validation split only by maximising ACC over a quantile grid.
    """
    res_g = np.asarray(res_g)
    res_i = np.asarray(res_i)
    all_res = np.concatenate([res_g, res_i])

    qs = np.linspace(0.01, 0.99, 199)
    grid = np.quantile(all_res, qs)

    best_tau = float(grid[0])
    best_metrics = metrics_at_tau(res_g, res_i, best_tau)
    best_acc = best_metrics["ACC"]

    for tau in grid[1:]:
        m = metrics_at_tau(res_g, res_i, float(tau))
        if m["ACC"] > best_acc:
            best_acc = m["ACC"]
            best_tau = float(tau)
            best_metrics = m

    return best_tau, best_metrics


# ---------------------------------------------------------------------
# Evaluation pipeline
# ---------------------------------------------------------------------


def evaluate_keyboard(
    val_csv: Path,
    test_csv: Path,
    model_root: Path,
    min_val_genuine: int = 20,
    max_impostor: int | None = None,
    random_state: int = 123,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Evaluate per-user CAE baseline models using explicit validation/test splits.

    Threshold tau is chosen on validation only.
    FAR/FRR/ACC are reported on held-out test only.

    Returns:
        df_metrics : per-user metrics
        macro      : macro-averaged metrics over users
        scores_df  : global score-level dataframe for plots
                     columns: ['label', 'score', 'user_id']
    """
    val_csv = Path(val_csv)
    test_csv = Path(test_csv)
    model_root = Path(model_root)

    val_df = _load_split(val_csv)
    test_df = _load_split(test_csv)

    users = sorted(set(val_df[LABEL_COL].unique()) & set(test_df[LABEL_COL].unique()))
    rows = []

    all_scores: list[float] = []
    all_labels: list[int] = []
    all_users: list[str] = []

    for user in users:
        user_dir = model_root / str(user)
        if not user_dir.is_dir():
            print(f"[WARN] Skipping user {user}: missing model directory {user_dir}")
            continue

        df_vg = val_df[val_df[LABEL_COL] == user]
        df_vi = val_df[val_df[LABEL_COL] != user]
        df_tg = test_df[test_df[LABEL_COL] == user]
        df_ti = test_df[test_df[LABEL_COL] != user]

        if len(df_vg) < min_val_genuine or len(df_vi) < 1 or len(df_tg) < 1 or len(df_ti) < 1:
            print(
                f"[WARN] Skipping user {user}: "
                f"val_genuine={len(df_vg)}, val_impostor={len(df_vi)}, "
                f"test_genuine={len(df_tg)}, test_impostor={len(df_ti)}"
            )
            continue

        n_val_i = min(len(df_vi), max_impostor or len(df_vg))
        n_test_i = min(len(df_ti), max_impostor or len(df_tg))

        df_vi_sample = df_vi.sample(n=n_val_i, random_state=random_state, replace=False)
        df_ti_sample = df_ti.sample(n=n_test_i, random_state=random_state, replace=False)

        X_vg = df_vg[FEATURE_COLS].to_numpy(dtype=np.float32)
        X_vi = df_vi_sample[FEATURE_COLS].to_numpy(dtype=np.float32)
        X_tg = df_tg[FEATURE_COLS].to_numpy(dtype=np.float32)
        X_ti = df_ti_sample[FEATURE_COLS].to_numpy(dtype=np.float32)

        scaler, model = _load_user_model_and_scaler(user_dir)

        res_vg = compute_residuals(X_vg, scaler, model)
        res_vi = compute_residuals(X_vi, scaler, model)
        tau, val_metrics = find_best_tau(res_vg, res_vi)

        res_tg = compute_residuals(X_tg, scaler, model)
        res_ti = compute_residuals(X_ti, scaler, model)
        test_metrics = metrics_at_tau(res_tg, res_ti, tau)
        err = 0.5 * (test_metrics["FAR"] + test_metrics["FRR"])

        rows.append(
            {
                "user_id": user,
                "best_tau": float(tau),
                "FAR": float(test_metrics["FAR"]),
                "FRR": float(test_metrics["FRR"]),
                "ACC": float(test_metrics["ACC"]),
                "ERR": float(err),
                "n_genuine": int(len(df_tg)),
                "n_impostor": int(len(df_ti_sample)),
                "val_FAR": float(val_metrics["FAR"]),
                "val_FRR": float(val_metrics["FRR"]),
                "val_ACC": float(val_metrics["ACC"]),
                "n_val_genuine": int(len(df_vg)),
                "n_val_impostor": int(len(df_vi_sample)),
            }
        )

        print(
            f"{str(user):>6s}\t"
            f"tau={tau:.6f}\t"
            f"FAR={test_metrics['FAR']:.3f}\t"
            f"FRR={test_metrics['FRR']:.3f}\t"
            f"ACC={test_metrics['ACC']:.3f}\t"
            f"n_test_g={len(df_tg):d}\t"
            f"n_test_i={len(df_ti_sample):d}"
        )

        # score = -residual (higher score = more genuine)
        all_scores.extend((-res_tg).tolist())
        all_labels.extend([1] * len(res_tg))
        all_users.extend([user] * len(res_tg))

        all_scores.extend((-res_ti).tolist())
        all_labels.extend([0] * len(res_ti))
        all_users.extend([user] * len(res_ti))

    if not rows:
        raise RuntimeError("No users had enough data to evaluate.")

    df_metrics = pd.DataFrame(rows)

    macro_far = float(df_metrics["FAR"].mean())
    macro_frr = float(df_metrics["FRR"].mean())
    macro_acc = float(df_metrics["ACC"].mean())
    macro_err = float(df_metrics["ERR"].mean())

    macro = {
        "MACRO_FAR": macro_far,
        "MACRO_FRR": macro_frr,
        "MACRO_ACC": macro_acc,
        "MACRO_ERR": macro_err,
        "n_users": int(len(df_metrics)),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "model_root": str(model_root),
        "model_type": "per_user_cae_baseline",
    }

    print()
    print(
        f"MACRO FAR={macro_far:.3f} "
        f"FRR={macro_frr:.3f} "
        f"ACC={macro_acc:.3f} "
        f"ERR={macro_err:.3f}"
    )

    scores_df = pd.DataFrame(
        {
            "label": all_labels,   # 1 = genuine, 0 = impostor
            "score": all_scores,   # higher = more genuine
            "user_id": all_users,
        }
    )

    return df_metrics, macro, scores_df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate per-user keyboard CAE baseline on explicit validation/test splits"
    )
    p.add_argument(
        "--val-csv",
        type=str,
        default=str(DEFAULT_VAL_CSV),
        help=f"Path to keyboard validation CSV (default: {DEFAULT_VAL_CSV})",
    )
    p.add_argument(
        "--test-csv",
        type=str,
        default=str(DEFAULT_TEST_CSV),
        help=f"Path to keyboard test CSV (default: {DEFAULT_TEST_CSV})",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=f"Directory containing per-user kb_cae/<user_id>/ models (default: {DEFAULT_MODEL_DIR})",
    )
    p.add_argument(
        "--min-val-genuine",
        type=int,
        default=20,
        help="Minimum validation genuine windows per user to include (default: 20)",
    )
    p.add_argument(
        "--max-impostor",
        type=int,
        default=None,
        help="Max impostor windows per user for validation/test (default: balance to genuine count)",
    )
    return p.parse_args()



def main() -> None:
    args = parse_args()
    val_csv = Path(args.val_csv)
    test_csv = Path(args.test_csv)
    model_dir = Path(args.model_dir)

    df_metrics, macro, scores_df = evaluate_keyboard(
        val_csv=val_csv,
        test_csv=test_csv,
        model_root=model_dir,
        min_val_genuine=args.min_val_genuine,
        max_impostor=args.max_impostor,
    )

    # Explicit outputs for the newer report path
    metrics_csv_exp = ARTIFACTS_DIR / "kb_cae_metrics_explicit.csv"
    overview_json_exp = ARTIFACTS_DIR / "kb_cae_overview_explicit.json"
    scores_csv_exp = ARTIFACTS_DIR / "kb_cae_scores_explicit.csv"

    df_metrics.to_csv(metrics_csv_exp, index=False)
    with overview_json_exp.open("w", encoding="utf-8") as fh:
        json.dump(macro, fh, indent=2)
    scores_df.to_csv(scores_csv_exp, index=False)

    # Legacy aliases for old plotting/report scripts
    metrics_csv_legacy = ARTIFACTS_DIR / "kb_cae_metrics.csv"
    overview_json_legacy = ARTIFACTS_DIR / "kb_cae_overview.json"
    scores_csv_legacy = ARTIFACTS_DIR / "kb_cae_scores.csv"

    df_metrics.to_csv(metrics_csv_legacy, index=False)
    with overview_json_legacy.open("w", encoding="utf-8") as fh:
        json.dump(macro, fh, indent=2)
    scores_df.to_csv(scores_csv_legacy, index=False)

    print()
    print(f"WROTE CAE metrics CSV -> {metrics_csv_exp}")
    print(f"WROTE CAE macro JSON  -> {overview_json_exp}")
    print(f"WROTE CAE scores CSV  -> {scores_csv_exp}")
    print(f"WROTE legacy metrics  -> {metrics_csv_legacy}")
    print(f"WROTE legacy overview -> {overview_json_legacy}")
    print(f"WROTE legacy scores   -> {scores_csv_legacy}")


if __name__ == "__main__":
    main()
