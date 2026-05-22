"""Build per-window keyboard CAE trust scores for genuine vs impostor plots.

This script is aligned with the newer per-user keyboard CAE baseline path.
It:
  - loads explicit validation/test splits
  - loads each user's CAE model from Models/kb_cae/<user_id>/
  - chooses tau on validation only
  - computes residuals on held-out test windows
  - writes a flat CSV for downstream plotting scripts:

      artifacts/plots/kb_cae_scores.csv

Columns written:
  claimed_user, user_id, session_id, window_id, trust, residual, score, label, best_tau

Where:
  - claimed_user = enrolled user whose CAE model is being evaluated
  - user_id      = actual source user of that row
  - label        = 1 for genuine, 0 for impostor
  - score        = -residual (higher is more genuine)
  - trust        = tau-relative trust in [0,1] for convenient plotting
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "Data"
MODEL_ROOT = BASE_DIR / "Models" / "kb_cae"
PLOTS_DIR = BASE_DIR / "artifacts" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VAL_CSV = DATA_DIR / "keyboard_val_windows.csv"
DEFAULT_TEST_CSV = DATA_DIR / "keyboard_test_windows.csv"
OUT_CSV = PLOTS_DIR / "kb_cae_scores.csv"

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


def _load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Keyboard CAE split not found at: {path}")

    df = pd.read_csv(path)
    needed = [LABEL_COL] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} missing columns: {missing}")

    extra = [c for c in ["session_id", "window_id"] if c in df.columns]
    df = df[needed + extra].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    return df


def _load_user_model_and_scaler(user_dir: Path):
    scaler_path = user_dir / "scaler.joblib"
    model_path = user_dir / "cae.keras"

    if not scaler_path.is_file():
        raise SystemExit(f"Missing scaler.joblib at: {scaler_path}")
    if not model_path.is_file():
        raise SystemExit(f"Missing cae.keras at: {model_path}")

    scaler = joblib_load(scaler_path)
    model = keras.models.load_model(model_path)
    return scaler, model


def _compute_residuals(X: np.ndarray, scaler, model: keras.Model) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled, batch_size=512, verbose=0)
    pred = np.asarray(pred, dtype=np.float32)

    if pred.shape != X_scaled.shape:
        raise RuntimeError(f"Prediction shape {pred.shape} != input shape {X_scaled.shape}")

    return np.mean((X_scaled - pred) ** 2, axis=1)


def _metrics_at_tau(res_g: np.ndarray, res_i: np.ndarray, tau: float) -> Dict[str, float]:
    tp = np.sum(res_g <= tau)
    fn = np.sum(res_g > tau)
    tn = np.sum(res_i > tau)
    fp = np.sum(res_i <= tau)

    n_g = len(res_g)
    n_i = len(res_i)

    far = fp / max(1, n_i)
    frr = fn / max(1, n_g)
    acc = (tp + tn) / max(1, n_g + n_i)
    return {"FAR": float(far), "FRR": float(frr), "ACC": float(acc)}


def _find_best_tau(res_g: np.ndarray, res_i: np.ndarray) -> Tuple[float, Dict[str, float]]:
    all_res = np.concatenate([res_g, res_i])
    grid = np.quantile(all_res, np.linspace(0.01, 0.99, 199))

    best_tau = float(grid[0])
    best_metrics = _metrics_at_tau(res_g, res_i, best_tau)
    best_acc = best_metrics["ACC"]

    for tau in grid[1:]:
        m = _metrics_at_tau(res_g, res_i, float(tau))
        if m["ACC"] > best_acc:
            best_acc = m["ACC"]
            best_tau = float(tau)
            best_metrics = m

    return best_tau, best_metrics


def _trust_from_residual(residual: float, tau: float) -> float:
    """
    Monotonic trust in [0,1] centred around tau.
    Residuals below tau -> trust above 0.5, above tau -> below 0.5.
    """
    scale = max(0.10 * max(tau, 1e-6), 1e-6)
    z = (float(tau) - float(residual)) / scale
    trust = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(trust, 0.0, 1.0))


def build_kb_cae_scores(
    val_csv: Path = DEFAULT_VAL_CSV,
    test_csv: Path = DEFAULT_TEST_CSV,
    min_val_genuine: int = 20,
    max_impostor: int | None = None,
    random_state: int = 123,
) -> Path:
    val_df = _load_split(Path(val_csv))
    test_df = _load_split(Path(test_csv))

    users = sorted(set(val_df[LABEL_COL].unique()) & set(test_df[LABEL_COL].unique()))
    rows: List[Dict[str, Any]] = []

    for user in users:
        user_dir = MODEL_ROOT / str(user)
        if not user_dir.is_dir():
            print(f"[WARN] Skipping user {user}: missing model dir {user_dir}")
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

        scaler, model = _load_user_model_and_scaler(user_dir)

        X_vg = df_vg[FEATURE_COLS].to_numpy(dtype=np.float32)
        X_vi = df_vi_sample[FEATURE_COLS].to_numpy(dtype=np.float32)
        res_vg = _compute_residuals(X_vg, scaler, model)
        res_vi = _compute_residuals(X_vi, scaler, model)
        tau, _ = _find_best_tau(res_vg, res_vi)

        X_tg = df_tg[FEATURE_COLS].to_numpy(dtype=np.float32)
        res_tg = _compute_residuals(X_tg, scaler, model)
        for idx, (_, row) in enumerate(df_tg.iterrows()):
            residual = float(res_tg[idx])
            rows.append(
                {
                    "claimed_user": str(user),
                    "user_id": str(row[LABEL_COL]),
                    "session_id": row.get("session_id", None),
                    "window_id": row.get("window_id", None),
                    "trust": _trust_from_residual(residual, tau),
                    "residual": residual,
                    "score": -residual,
                    "label": 1,
                    "best_tau": float(tau),
                }
            )

        X_ti = df_ti_sample[FEATURE_COLS].to_numpy(dtype=np.float32)
        res_ti = _compute_residuals(X_ti, scaler, model)
        for idx, (_, row) in enumerate(df_ti_sample.iterrows()):
            residual = float(res_ti[idx])
            rows.append(
                {
                    "claimed_user": str(user),
                    "user_id": str(row[LABEL_COL]),
                    "session_id": row.get("session_id", None),
                    "window_id": row.get("window_id", None),
                    "trust": _trust_from_residual(residual, tau),
                    "residual": residual,
                    "score": -residual,
                    "label": 0,
                    "best_tau": float(tau),
                }
            )

    if not rows:
        raise SystemExit("No rows collected; check data, models, and thresholds.")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"[OK] Wrote keyboard CAE scores to: {OUT_CSV}")
    print(
        f"[INFO] Total samples: {len(out_df)} | "
        f"genuine={int((out_df['label'] == 1).sum())} | "
        f"impostor={int((out_df['label'] == 0).sum())}"
    )
    print(f"[INFO] Users covered: {out_df['claimed_user'].nunique()}")
    return OUT_CSV


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-window keyboard CAE trust scores for plotting.",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=str(DEFAULT_VAL_CSV),
        help="Keyboard validation CSV (default: Data/keyboard_val_windows.csv).",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(DEFAULT_TEST_CSV),
        help="Keyboard test CSV (default: Data/keyboard_test_windows.csv).",
    )
    parser.add_argument(
        "--min-val-genuine",
        type=int,
        default=20,
        help="Minimum validation genuine windows per user.",
    )
    parser.add_argument(
        "--max-impostor",
        type=int,
        default=None,
        help="Maximum impostor windows per user (default: balance to genuine count).",
    )
    args = parser.parse_args()

    build_kb_cae_scores(
        val_csv=Path(args.val_csv),
        test_csv=Path(args.test_csv),
        min_val_genuine=int(args.min_val_genuine),
        max_impostor=int(args.max_impostor) if args.max_impostor is not None else None,
    )


if __name__ == "__main__":
    main()