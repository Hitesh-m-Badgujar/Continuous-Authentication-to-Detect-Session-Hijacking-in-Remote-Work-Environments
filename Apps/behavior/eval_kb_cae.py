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
MODELS_DIR = BASE_DIR / "Models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# IMPORTANT: use CMU-derived windows
DEFAULT_CSV = DATA_DIR / "kb_cmu_windows.csv"
DEFAULT_MODEL_DIR = MODELS_DIR / "cae_kb"

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

def load_model_and_scaler(model_dir: Path) -> Tuple[object, keras.Model]:
    model_dir = Path(model_dir)
    scaler_path = model_dir / "scaler.joblib"
    model_path = model_dir / "cae.keras"

    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing scaler.joblib at {scaler_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing cae.keras at {model_path}")

    scaler = joblib_load(scaler_path)
    model = keras.models.load_model(model_path)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        x_shape = input_shape[0]
    else:
        x_shape = input_shape

    if len(x_shape) != 2 or (x_shape[1] is not None and x_shape[1] != len(FEATURE_COLS)):
        print(
            f"[WARN] Model input_shape={input_shape} does not match "
            f"{len(FEATURE_COLS)}-D FEATURE_COLS"
        )

    return scaler, model


def compute_residuals(
    X: np.ndarray,
    scaler,
    model: keras.Model,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X_scaled = scaler.transform(X)

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        n_inputs = len(input_shape)
    else:
        n_inputs = 1

    if n_inputs == 1:
        pred = model.predict(X_scaled, batch_size=512, verbose=0)
    elif n_inputs == 2:
        x_shape, cond_shape = input_shape
        cond_dim = 1
        if cond_shape is not None and len(cond_shape) >= 2 and cond_shape[1] is not None:
            cond_dim = int(cond_shape[1])
        cond = np.zeros((X_scaled.shape[0], cond_dim), dtype="int32")
        pred = model.predict([X_scaled, cond], batch_size=512, verbose=0)
    else:
        raise RuntimeError(f"Unsupported number of model inputs: {n_inputs}")

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
    Search over candidate taus (quantiles of all residuals) and find
    the tau that MAXIMISES ACC (not EER).
    """
    res_g = np.asarray(res_g)
    res_i = np.asarray(res_i)
    all_res = np.concatenate([res_g, res_i])

    # Grid of candidate thresholds
    qs = np.linspace(0.05, 0.95, 181)
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
    csv_path: Path,
    model_dir: Path,
    min_genuine: int = 200,
    max_impostor: int | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Main evaluation routine.

    - min_genuine: minimum genuine windows per user
    - max_impostor: if None, we use the same number as genuine (balanced)

    Returns:
        df_metrics : per-user FAR/FRR/ACC etc.
        macro      : macro-averaged metrics
        scores_df  : global score-level dataframe for confusion/PR
                     columns: ['label', 'score', 'user_id']
    """
    csv_path = Path(csv_path)
    model_dir = Path(model_dir)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Keyboard windows CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if "user_id" not in df.columns:
        raise ValueError(f"CSV {csv_path} must contain a 'user_id' column")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing feature columns: {missing}")

    scaler, model = load_model_and_scaler(model_dir)

    users = sorted(df["user_id"].unique())
    rows = []

    all_scores: list[float] = []
    all_labels: list[int] = []
    all_users: list[str] = []

    for user in users:
        df_g = df[df["user_id"] == user]
        df_i = df[df["user_id"] != user]

        if len(df_g) < min_genuine or len(df_i) < min_genuine:
            print(
                f"[WARN] Skipping user {user}: "
                f"n_genuine={len(df_g)}, n_impostor={len(df_i)} "
                f"(need >= {min_genuine} of each)"
            )
            continue

        n_g = min(len(df_g), min_genuine)
        n_i = min(len(df_i), max_impostor or n_g)

        df_g_sample = df_g.sample(n=n_g, random_state=123, replace=False)
        df_i_sample = df_i.sample(n=n_i, random_state=123, replace=False)

        Xg = df_g_sample[FEATURE_COLS].to_numpy()
        Xi = df_i_sample[FEATURE_COLS].to_numpy()

        res_g = compute_residuals(Xg, scaler, model)
        res_i = compute_residuals(Xi, scaler, model)

        tau, m = find_best_tau(res_g, res_i)
        err = 0.5 * (m["FAR"] + m["FRR"])

        rows.append(
            {
                "user_id": user,
                "best_tau": tau,
                "FAR": m["FAR"],
                "FRR": m["FRR"],
                "ACC": m["ACC"],
                "ERR": err,
                "n_genuine": n_g,
                "n_impostor": n_i,
            }
        )

        print(
            f"{str(user):>6s}\t"
            f"tau={tau:.6f}\t"
            f"FAR={m['FAR']:.3f}\t"
            f"FRR={m['FRR']:.3f}\t"
            f"ACC={m['ACC']:.3f}\t"
            f"n_g={n_g:d}\t"
            f"n_i={n_i:d}"
        )

        # ------------- accumulate score-level data (for confusion/PR) -------------
        # score = -residual  (higher score = more genuine)
        all_scores.extend((-res_g).tolist())
        all_labels.extend([1] * len(res_g))   # genuine
        all_users.extend([user] * len(res_g))

        all_scores.extend((-res_i).tolist())
        all_labels.extend([0] * len(res_i))   # impostor
        all_users.extend([user] * len(res_i))

    if not rows:
        raise RuntimeError("No users had enough data to evaluate.")

    df_metrics = pd.DataFrame(rows)

    macro_far = float(df_metrics["FAR"].mean())
    macro_frr = float(df_metrics["FRR"].mean())
    macro_acc = float(df_metrics["ACC"].mean())

    macro = {
        "MACRO_FAR": macro_far,
        "MACRO_FRR": macro_frr,
        "MACRO_ACC": macro_acc,
    }

    print()
    print(
        f"MACRO FAR={macro_far:.3f} "
        f"FRR={macro_frr:.3f} "
        f"ACC={macro_acc:.3f}"
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
        description="Evaluate keyboard CAE model on kb_cmu_windows.csv"
    )
    p.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help=f"Path to keyboard windows CSV (default: {DEFAULT_CSV})",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help=f"Directory containing cae.keras + scaler.joblib (default: {DEFAULT_MODEL_DIR})",
    )
    p.add_argument(
        "--min-genuine",
        type=int,
        default=200,
        help="Min genuine windows per user to include (default: 200)",
    )
    p.add_argument(
        "--max-impostor",
        type=int,
        default=None,
        help="Max impostor windows per user (default: same as genuine, balanced)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    model_dir = Path(args.model_dir)

    df_metrics, macro, scores_df = evaluate_keyboard(
        csv_path=csv_path,
        model_dir=model_dir,
        min_genuine=args.min_genuine,
        max_impostor=args.max_impostor,
    )

    metrics_csv = ARTIFACTS_DIR / "kb_cae_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)

    overview_json = ARTIFACTS_DIR / "kb_cae_overview.json"
    with overview_json.open("w", encoding="utf-8") as fh:
        json.dump(macro, fh, indent=2)

    scores_csv = ARTIFACTS_DIR / "kb_cae_scores.csv"
    scores_df.to_csv(scores_csv, index=False)

    print()
    print(f"WROTE CAE metrics CSV -> {metrics_csv}")
    print(f"WROTE CAE macro JSON  -> {overview_json}")
    print(f"WROTE CAE scores CSV  -> {scores_csv}")


if __name__ == "__main__":
    main()
