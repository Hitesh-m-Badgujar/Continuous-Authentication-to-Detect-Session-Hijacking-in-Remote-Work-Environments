from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# IMPORTANT: use CMU-derived windows
DATA_CSV = Path("Data/kb_cmu_windows.csv")
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
TRAIN_FRACTION = 0.5
N_IMPOSTORS = 3000


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

    print(f"[INFO] Loaded {len(df)} rows for {df['user_id'].nunique()} users")
    return df


def _scaled_manhattan_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    aad = np.mean(np.abs(X - mu), axis=0)
    aad[aad == 0.0] = 1e-6
    return mu, aad


def _distances_scaled_manhattan(X: np.ndarray, mu: np.ndarray, aad: np.ndarray) -> np.ndarray:
    diff = np.abs(X - mu)
    return np.sum(diff / aad, axis=1)


def eval_user(
    df_user: pd.DataFrame,
    df_all: pd.DataFrame,
    features: List[str],
    train_fraction: float,
    n_impostors: int,
) -> Tuple[float, float, float, int, int]:
    sort_cols = [c for c in ["session_id", "start_idx", "window_id"] if c in df_user.columns]
    if sort_cols:
        df_user = df_user.sort_values(sort_cols)
    else:
        df_user = df_user.sort_index()

    n_total = len(df_user)
    n_train = max(1, int(n_total * train_fraction))
    if n_total - n_train < 1:
        raise ValueError("User has no test samples after split.")

    train_df = df_user.iloc[:n_train]
    test_df = df_user.iloc[n_train:]

    X_train = train_df[features].to_numpy(dtype=float)
    Xg = test_df[features].to_numpy(dtype=float)

    imp_pool = df_all[df_all["user_id"] != df_user["user_id"].iloc[0]]
    if len(imp_pool) > n_impostors:
        imp_df = imp_pool.sample(n=n_impostors, random_state=42)
    else:
        imp_df = imp_pool
    Xi = imp_df[features].to_numpy(dtype=float)

    mu, aad = _scaled_manhattan_stats(X_train)
    dg = _distances_scaled_manhattan(Xg, mu, aad)
    di = _distances_scaled_manhattan(Xi, mu, aad)

    taus = np.unique(np.concatenate([dg, di]))
    best_ACC = -1.0
    best_FAR = 1.0
    best_FRR = 1.0

    for tau in taus:
        FAR = float((di <= tau).mean())
        FRR = float((dg > tau).mean())
        ACC = 1.0 - 0.5 * (FAR + FRR)
        if ACC > best_ACC:
            best_ACC = ACC
            best_FAR = FAR
            best_FRR = FRR

    return best_FAR, best_FRR, best_ACC, len(Xg), len(Xi)


def main() -> None:
    print("[INFO] eval_kb_manhattan starting")
    df = _load_data()
    users = sorted(df["user_id"].unique())
    print(f"[INFO] Evaluating {len(users)} users")

    rows = []
    for uid in users:
        df_u = df[df["user_id"] == uid].copy()
        if len(df_u) < MIN_PER_USER:
            continue
        FAR, FRR, ACC, n_g, n_i = eval_user(
            df_u,
            df,
            FEATURES,
            train_fraction=TRAIN_FRACTION,
            n_impostors=N_IMPOSTORS,
        )
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
    metrics_csv = ARTIFACTS_DIR / "kb_manhattan_eval_metrics.csv"
    overview_json = ARTIFACTS_DIR / "kb_manhattan_eval_overview.json"

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
        "features": FEATURES,
        "detector": "scaled_manhattan",
    }
    import json

    with open(overview_json, "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)

    print(f"\nWROTE Manhattan metrics CSV -> {metrics_csv}")
    print(f"WROTE Manhattan macro JSON  -> {overview_json}")


if __name__ == "__main__":
    main()
