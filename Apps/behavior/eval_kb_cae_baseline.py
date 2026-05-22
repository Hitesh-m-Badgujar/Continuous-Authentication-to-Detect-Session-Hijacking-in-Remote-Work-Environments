from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

# TensorFlow / Keras
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
try:
    from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
except Exception:
    LegacyAdam = Adam

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
ART_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "Models" / "kb_cae"

TRAIN_CSV = DATA_DIR / "keyboard_train_windows.csv"
VAL_CSV = DATA_DIR / "keyboard_val_windows.csv"
TEST_CSV = DATA_DIR / "keyboard_test_windows.csv"

FEATURE_COLS: List[str] = [
    "dwell_mean", "dwell_std", "dwell_p10", "dwell_p50", "dwell_p90",
    "dd_mean", "dd_std", "dd_p10", "dd_p50", "dd_p90",
    "ud_mean", "ud_std", "ud_p10", "ud_p50", "ud_p90",
    "backspace_rate", "burst_mean", "idle_frac",
]

LABEL_COL = "user_id"
MIN_TRAIN_GENUINE = 50
EPOCHS = 40
BATCH_SIZE = 256


def _load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)

    needed = [LABEL_COL] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = df[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(str)
    return df


def _filter_common_users(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    train_counts = train_df[LABEL_COL].value_counts()
    val_counts = val_df[LABEL_COL].value_counts()
    test_counts = test_df[LABEL_COL].value_counts()

    users = set(train_counts[train_counts >= MIN_TRAIN_GENUINE].index)
    users &= set(val_counts[val_counts >= 1].index)
    users &= set(test_counts[test_counts >= 1].index)

    train_df = train_df[train_df[LABEL_COL].isin(users)].copy()
    val_df = val_df[val_df[LABEL_COL].isin(users)].copy()
    test_df = test_df[test_df[LABEL_COL].isin(users)].copy()
    return train_df, val_df, test_df


def _build_autoencoder(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,), name="inp")
    x = Dense(64, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    z = Dense(16, activation="relu", name="bottleneck")(x)

    x = Dense(32, activation="relu")(z)
    x = Dense(64, activation="relu")(x)
    out = Dense(input_dim, activation="linear")(x)

    model = Model(inp, out, name="kb_cae_baseline")
    model.compile(optimizer=LegacyAdam(learning_rate=1e-3), loss="mse")
    return model


def _residuals(model: Model, X: np.ndarray) -> np.ndarray:
    pred = model.predict(X, verbose=0)
    return np.mean((X - pred) ** 2, axis=1)


def _select_tau(genuine_res: np.ndarray, impostor_res: np.ndarray) -> Tuple[float, float, float, float]:
    scores = np.concatenate([genuine_res, impostor_res])
    if len(scores) == 0:
        return 0.0, 1.0, 1.0, 0.0

    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi <= lo:
        tau = lo
        pred_gen = genuine_res <= tau
        pred_imp = impostor_res <= tau
        far = float(np.mean(pred_imp)) if len(impostor_res) else 0.0
        frr = 1.0 - float(np.mean(pred_gen)) if len(genuine_res) else 0.0
        acc = (np.sum(pred_gen) + np.sum(~pred_imp)) / (len(genuine_res) + len(impostor_res))
        return tau, far, frr, float(acc)

    taus = np.linspace(lo, hi, 200)
    best = None

    for tau in taus:
        pred_gen = genuine_res <= tau
        pred_imp = impostor_res <= tau

        far = float(np.mean(pred_imp)) if len(impostor_res) else 0.0
        frr = 1.0 - float(np.mean(pred_gen)) if len(genuine_res) else 0.0
        acc = (np.sum(pred_gen) + np.sum(~pred_imp)) / (len(genuine_res) + len(impostor_res))

        if best is None or acc > best[3]:
            best = (float(tau), far, frr, float(acc))

    return best


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = _load_split(TRAIN_CSV)
    val_df = _load_split(VAL_CSV)
    test_df = _load_split(TEST_CSV)
    train_df, val_df, test_df = _filter_common_users(train_df, val_df, test_df)

    users = sorted(train_df[LABEL_COL].unique())
    print(f"[INFO] CAE baseline users retained: {len(users)}")

    rows = []

    for user in users:
        print(f"\n[INFO] Training CAE for {user}")

        tr_g = train_df[train_df[LABEL_COL] == user][FEATURE_COLS].to_numpy(dtype=float)
        va_g = val_df[val_df[LABEL_COL] == user][FEATURE_COLS].to_numpy(dtype=float)
        va_i = val_df[val_df[LABEL_COL] != user][FEATURE_COLS].to_numpy(dtype=float)
        te_g = test_df[test_df[LABEL_COL] == user][FEATURE_COLS].to_numpy(dtype=float)
        te_i = test_df[test_df[LABEL_COL] != user][FEATURE_COLS].to_numpy(dtype=float)

        scaler = StandardScaler()
        tr_g_s = scaler.fit_transform(tr_g)
        va_g_s = scaler.transform(va_g)
        va_i_s = scaler.transform(va_i)
        te_g_s = scaler.transform(te_g)
        te_i_s = scaler.transform(te_i)

        model = _build_autoencoder(input_dim=len(FEATURE_COLS))
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        model.fit(
            tr_g_s, tr_g_s,
            validation_data=(va_g_s, va_g_s),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[es],
            verbose=0,
        )

        val_g_res = _residuals(model, va_g_s)
        val_i_res = _residuals(model, va_i_s)
        tau, val_far, val_frr, val_acc = _select_tau(val_g_res, val_i_res)

        test_g_res = _residuals(model, te_g_s)
        test_i_res = _residuals(model, te_i_s)

        pred_gen = test_g_res <= tau
        pred_imp = test_i_res <= tau

        far = float(np.mean(pred_imp)) if len(test_i_res) else 0.0
        frr = 1.0 - float(np.mean(pred_gen)) if len(test_g_res) else 0.0
        acc = (np.sum(pred_gen) + np.sum(~pred_imp)) / (len(test_g_res) + len(test_i_res))
        err = 0.5 * (far + frr)

        user_dir = MODELS_DIR / user
        user_dir.mkdir(parents=True, exist_ok=True)
        model.save(user_dir / "cae.keras")
        dump(scaler, user_dir / "scaler.joblib")
        with open(user_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"t₹au": tau, "user_id": user}, f, indent=2)

        rows.append(
            {
                "user_id": user,
                "best_tau": float(tau),
                "FAR": float(far),
                "FRR": float(frr),
                "ACC": float(acc),
                "ERR": float(err),
                "n_genuine": int(len(test_g_res)),
                "n_impostor": int(len(test_i_res)),
                "val_FAR": float(val_far),
                "val_FRR": float(val_frr),
                "val_ACC": float(val_acc),
            }
        )

        print(
            f"  tau={tau:.6f}  FAR={far:.3f}  FRR={frr:.3f}  "
            f"ACC={acc:.3f}  ERR={err:.3f}"
        )

    metrics_df = pd.DataFrame(rows)
    macro_far = float(metrics_df["FAR"].mean()) if len(metrics_df) else 0.0
    macro_frr = float(metrics_df["FRR"].mean()) if len(metrics_df) else 0.0
    macro_acc = float(metrics_df["ACC"].mean()) if len(metrics_df) else 0.0
    macro_err = float(metrics_df["ERR"].mean()) if len(metrics_df) else 0.0

    metrics_csv = ART_DIR / "kb_cae_metrics_explicit.csv"
    overview_json = ART_DIR / "kb_cae_overview_explicit.json"

    metrics_df.to_csv(metrics_csv, index=False)
    with open(overview_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "macro_far": macro_far,
                "macro_frr": macro_frr,
                "macro_acc": macro_acc,
                "macro_err": macro_err,
                "n_users": len(users),
                "train_csv": str(TRAIN_CSV),
                "val_csv": str(VAL_CSV),
                "test_csv": str(TEST_CSV),
                "feature_cols": FEATURE_COLS,
                "model": "per-user conditional autoencoder baseline",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
            },
            f,
            indent=2,
        )

    print("\n[INFO] Wrote CAE baseline outputs:")
    print(f"  {metrics_csv}")
    print(f"  {overview_json}")
    print(
        f"[INFO] Macro FAR={macro_far:.3f}  Macro FRR={macro_frr:.3f}  "
        f"Macro ACC={macro_acc:.3f}  Macro ERR={macro_err:.3f}"
    )


if __name__ == "__main__":
    main()