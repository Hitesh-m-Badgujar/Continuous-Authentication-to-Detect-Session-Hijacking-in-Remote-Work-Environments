# Apps/behavior/train_kb_cae_cmu.py
"""
Train per-user keyboard CAE baseline models using explicit train/validation splits.

This file is kept only for the keyboard CAE baseline path, not for the final
runtime keyboard model. The final runtime keyboard path is SVM-based.

Inputs:
  Data/keyboard_train_windows.csv
  Data/keyboard_val_windows.csv

Outputs:
  Models/kb_cae/<user_id>/scaler.joblib
  Models/kb_cae/<user_id>/cae.keras
  Models/kb_cae/<user_id>/meta.json
  artifacts/keyboard/kb_cae_train_summary.csv
  artifacts/keyboard/kb_cae_train_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
try:
    from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam
except Exception:
    LegacyAdam = keras.optimizers.Adam

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models" / "kb_cae"
ART_DIR = BASE_DIR / "artifacts" / "keyboard"

TRAIN_CSV = DATA_DIR / "keyboard_train_windows.csv"
VAL_CSV = DATA_DIR / "keyboard_val_windows.csv"

LABEL_COL = "user_id"
MIN_TRAIN_GENUINE = 50
EPOCHS = 40
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.2

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


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing keyboard CAE split: {path}")

    df = pd.read_csv(path)
    needed = [LABEL_COL] + FEATURES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df = df[needed].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    return df


def _filter_common_users(train_df: pd.DataFrame, val_df: pd.DataFrame):
    train_counts = train_df[LABEL_COL].value_counts()
    val_counts = val_df[LABEL_COL].value_counts()

    keep = set(train_counts[train_counts >= MIN_TRAIN_GENUINE].index)
    keep &= set(val_counts[val_counts >= 1].index)

    train_df = train_df[train_df[LABEL_COL].isin(keep)].copy()
    val_df = val_df[val_df[LABEL_COL].isin(keep)].copy()
    if train_df.empty:
        raise RuntimeError("No keyboard CAE users remain after filtering")
    return train_df, val_df


def build_cae(input_dim: int) -> keras.Model:
    inp = keras.Input(shape=(input_dim,), name="kb_features")

    x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(1e-4))(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)

    x = keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    bottleneck = keras.layers.Dense(
        16,
        activation="relu",
        name="bottleneck",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)

    x = keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(1e-4))(bottleneck)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(DROPOUT_RATE)(x)

    out = keras.layers.Dense(input_dim, activation=None, name="recon")(x)

    model = keras.Model(inputs=inp, outputs=out, name="kb_cae_baseline")
    model.compile(optimizer=LegacyAdam(learning_rate=LEARNING_RATE), loss="mse")
    return model


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    train_df = _load_split(TRAIN_CSV)
    val_df = _load_split(VAL_CSV)
    train_df, val_df = _filter_common_users(train_df, val_df)

    users = sorted(train_df[LABEL_COL].unique())
    print(f"[INFO] Keyboard CAE users retained: {len(users)}")

    summary_rows = []

    for user in users:
        user_train = train_df[train_df[LABEL_COL] == user][FEATURES].to_numpy(dtype=np.float32)
        user_val = val_df[val_df[LABEL_COL] == user][FEATURES].to_numpy(dtype=np.float32)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(user_train)
        X_val = scaler.transform(user_val)

        model = build_cae(X_train.shape[1])
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=0,
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=0,
        )

        history = model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            verbose=0,
            callbacks=[early_stop, reduce_lr],
        )

        user_dir = MODELS_DIR / user
        user_dir.mkdir(parents=True, exist_ok=True)

        scaler_path = user_dir / "scaler.joblib"
        model_path = user_dir / "cae.keras"
        meta_path = user_dir / "meta.json"

        dump(scaler, scaler_path)
        model.save(model_path)

        best_val_loss = float(np.min(history.history["val_loss"])) if history.history.get("val_loss") else None
        epochs_run = int(len(history.history.get("loss", [])))

        meta = {
            "user_id": user,
            "features": FEATURES,
            "input_dim": int(X_train.shape[1]),
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "epochs_run": epochs_run,
            "best_val_loss": best_val_loss,
            "train_csv": str(TRAIN_CSV),
            "val_csv": str(VAL_CSV),
            "model_type": "per_user_cae_baseline",
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        summary_rows.append(meta)
        print(
            f"[INFO] Trained CAE for {user}: n_train={len(X_train)}, n_val={len(X_val)}, "
            f"epochs={epochs_run}, best_val_loss={best_val_loss:.6f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = ART_DIR / "kb_cae_train_summary.csv"
    summary_json = ART_DIR / "kb_cae_train_summary.json"

    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary_rows, fh, indent=2)

    print(f"[INFO] Saved keyboard CAE summary -> {summary_csv}")
    print(f"[INFO] Saved keyboard CAE summary -> {summary_json}")
    print("[INFO] Keyboard CAE baseline training complete.")


if __name__ == "__main__":
    main()