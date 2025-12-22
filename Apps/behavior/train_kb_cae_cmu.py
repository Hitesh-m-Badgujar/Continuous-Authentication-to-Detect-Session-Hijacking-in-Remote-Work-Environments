# Apps/behavior/train_kb_cae_cmu.py
"""
Train a stronger keyboard CAE on CMU-derived windows (kb_cmu_windows.csv).

Outputs:
  Models/cae_kb/scaler.joblib
  Models/cae_kb/cae.keras
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import dump
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models" / "cae_kb"

DATA_CSV = DATA_DIR / "kb_cmu_windows.csv"

# Users with fewer windows than this are dropped
MIN_PER_USER = 40

# Training hyperparameters
EPOCHS = 120           # upper bound, EarlyStopping will usually stop earlier
BATCH_SIZE = 512
VAL_SPLIT = 0.1
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

def load_cmu_windows() -> pd.DataFrame:
    """Load kb_cmu_windows.csv and filter users with enough data."""
    if not DATA_CSV.is_file():
        raise FileNotFoundError(f"Keyboard CMU windows CSV not found at: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    if "user_id" not in df.columns:
        raise ValueError(f"{DATA_CSV} must contain 'user_id' column")

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{DATA_CSV} is missing feature columns: {missing}")

    df["user_id"] = df["user_id"].astype(str).str.strip()

    vc = df["user_id"].value_counts()
    keep = vc[vc >= MIN_PER_USER].index
    df = df[df["user_id"].isin(keep)].copy()

    if df.empty:
        raise RuntimeError("No users with enough windows after MIN_PER_USER filtering")

    print(
        f"[INFO] Loaded {len(df)} windows for {df['user_id'].nunique()} users "
        f"from {DATA_CSV}"
    )
    return df


def build_strong_cae(input_dim: int) -> keras.Model:
    """
    Stronger fully-connected CAE:

        input (18)
          -> Dense(64) + BN + ReLU + Dropout
          -> Dense(32) + BN + ReLU
          -> Dense(16) bottleneck
          -> Dense(32) + BN + ReLU
          -> Dense(64) + BN + ReLU + Dropout
          -> Dense(18) reconstruction

    Optimiser: Adam with small LR
    Loss: MSE
    """
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

    model = keras.Model(inputs=inp, outputs=out, name="kb_cae_strong")

    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss="mse")

    model.summary()
    return model


# ---------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_cmu_windows()
    X = df[FEATURES].to_numpy(dtype=np.float32)

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]
    print(
        f"[INFO] Training STRONG CAE with input_dim={input_dim}, "
        f"samples={X_scaled.shape[0]}"
    )

    model = build_strong_cae(input_dim)

    # Callbacks: EarlyStopping + LR reduction
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1,
    )

    history = model.fit(
        X_scaled,
        X_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        shuffle=True,
        verbose=1,
        callbacks=[early_stop, reduce_lr],
    )

    # Save scaler + model
    scaler_path = MODELS_DIR / "scaler.joblib"
    model_path = MODELS_DIR / "cae.keras"

    dump(scaler, scaler_path)
    model.save(model_path)

    print(f"[INFO] Saved scaler -> {scaler_path}")
    print(f"[INFO] Saved CAE model -> {model_path}")
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
