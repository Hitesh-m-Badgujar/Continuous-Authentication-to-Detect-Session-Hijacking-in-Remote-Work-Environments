# Apps/behavior/Plots/keyboard/train_eval_kb_hmm.py
"""
Train and evaluate per-user HMMs for keyboard windows with basic hyperparameter tuning.

Goal:
  - Use kb_cmu_windows.csv (CMU dataset)
  - Build one GaussianHMM per user on window-level feature sequences
  - Tune n_states per user over a small grid [2, 3, 4] using train log-likelihood
  - Evaluate session-level classification accuracy on held-out sessions

This is for experimentation only – to compare HMM vs SVM accuracy for the thesis.
It does NOT touch your existing real-time SVM model files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

# hmmlearn is not part of scikit-learn; you MUST install it:
#   pip install hmmlearn
from hmmlearn.hmm import GaussianHMM


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
# H1/Apps/behavior/Plots/keyboard/train_eval_kb_hmm.py
# parents[0] = .../Plots/keyboard
# parents[1] = .../behavior/Plots
# parents[2] = .../behavior
# parents[3] = .../Apps
# parents[4] = .../H1
BASE_DIR = THIS_FILE.parents[4]

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import RuntimeScorer feature definitions (to keep feature order consistent)
from Apps.behavior import ae_conditional  # type: ignore[attr-defined]

FEATURE_COLS: List[str] = list(ae_conditional.FEATURE_COLS)

DATA_DIR = BASE_DIR / "Data"
KB_CSV = DATA_DIR / "kb_cmu_windows.csv"

MODELS_DIR = BASE_DIR / "Models" / "kb_hmm"
ARTIFACTS_DIR = BASE_DIR / "artifacts" / "keyboard"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Candidate number of states per HMM
N_STATES_GRID = [2, 3, 4]


# ---------------------------------------------------------------------
# Data loading / sequence building
# ---------------------------------------------------------------------

def load_kb_df() -> pd.DataFrame:
    if not KB_CSV.is_file():
        raise SystemExit(f"[FATAL] Keyboard CSV not found: {KB_CSV}")

    df = pd.read_csv(KB_CSV)
    # Basic checks
    for col in ("user_id", "session_id", "window_id"):
        if col not in df.columns:
            raise SystemExit(f"[FATAL] kb_cmu_windows.csv missing column '{col}'")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"[FATAL] kb_cmu_windows.csv missing feature columns: {missing}")

    df = df.dropna(subset=FEATURE_COLS).copy()
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["session_id"] = df["session_id"].astype(int)
    df["window_id"] = df["window_id"].astype(int)

    return df


def build_sequences(
    df: pd.DataFrame,
) -> Tuple[Dict[str, Dict[str, List[np.ndarray]]], StandardScaler]:
    """
    Build per-user sequences and a global scaler.

    Returns:
      - sequences: dict[user_id]["train" or "test"] = list of sequences (ndarray [T, D])
      - scaler   : fitted StandardScaler on ALL training windows
    """
    rng = np.random.RandomState(42)

    users = sorted(df["user_id"].unique())
    print(f"[INFO] Found {len(users)} users in kb_cmu_windows.csv")

    # Train/test split at session level per user
    per_user_splits: Dict[str, Dict[str, List[int]]] = {}

    for uid in users:
        sub = df[df["user_id"] == uid]
        sessions = sorted(sub["session_id"].unique())
        if len(sessions) < 2:
            raise SystemExit(
                f"[FATAL] User {uid} has < 2 sessions; cannot build train/test split."
            )

        # 75% train, 25% test at session level
        n_train = max(1, int(round(0.75 * len(sessions))))
        # ensure at least 1 test
        if n_train >= len(sessions):
            n_train = len(sessions) - 1

        train_sessions = sorted(rng.choice(sessions, size=n_train, replace=False))
        test_sessions = [s for s in sessions if s not in train_sessions]

        per_user_splits[uid] = {
            "train": train_sessions,
            "test": test_sessions,
        }

    # Build global training matrix for scaler
    X_train_all = []

    for uid in users:
        rows = df[(df["user_id"] == uid) & (df["session_id"].isin(per_user_splits[uid]["train"]))]
        X_train_all.append(rows[FEATURE_COLS].to_numpy(dtype=float))

    if not X_train_all:
        raise SystemExit("[FATAL] No training data collected when building scaler.")

    X_train_all_mat = np.vstack(X_train_all)
    scaler = StandardScaler()
    scaler.fit(X_train_all_mat)
    del X_train_all_mat

    # Now build sequences per user, using the scaler
    sequences: Dict[str, Dict[str, List[np.ndarray]]] = {uid: {"train": [], "test": []} for uid in users}

    for uid in users:
        user_rows = df[df["user_id"] == uid].copy()

        for split_name in ("train", "test"):
            for sess in per_user_splits[uid][split_name]:
                sess_rows = user_rows[user_rows["session_id"] == sess].sort_values("window_id")
                if sess_rows.empty:
                    continue
                X_seq = scaler.transform(sess_rows[FEATURE_COLS].to_numpy(dtype=float))
                sequences[uid][split_name].append(X_seq)

    print("[INFO] Sequences built.")
    return sequences, scaler


# ---------------------------------------------------------------------
# HMM training with per-user n_states tuning
# ---------------------------------------------------------------------

def _fit_hmm_for_user(
    uid: str,
    train_seqs: List[np.ndarray],
    n_states: int,
    covariance_type: str = "diag",
    n_iter: int = 40,
) -> Tuple[GaussianHMM, float]:
    """
    Fit a single HMM for one user for a given n_states and return
    (model, mean_train_loglik_per_frame).
    """
    # Concatenate all sequences with lengths
    X_list = []
    lengths = []
    for seq in train_seqs:
        seq = np.asarray(seq, dtype=float)
        if seq.ndim != 2:
            raise ValueError(f"Bad sequence shape for user {uid}: {seq.shape}")
        X_list.append(seq)
        lengths.append(len(seq))

    X_all = np.vstack(X_list)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=42,
        verbose=False,
    )
    model.fit(X_all, lengths)

    # Compute mean log-likelihood per frame over training sessions
    total_ll = 0.0
    total_T = 0
    for seq in train_seqs:
        seq = np.asarray(seq, dtype=float)
        ll = model.score(seq)
        total_ll += ll
        total_T += seq.shape[0]

    mean_ll = total_ll / max(total_T, 1)

    return model, mean_ll


def train_hmms_with_tuning(
    sequences: Dict[str, Dict[str, List[np.ndarray]]],
    n_states_grid: List[int],
    covariance_type: str = "diag",
    n_iter: int = 40,
) -> Tuple[Dict[str, GaussianHMM], Dict[str, int]]:
    """
    Train one GaussianHMM per user with a small grid search over n_states.

    For each user:
      - For each n_states in n_states_grid:
          * Fit HMM
          * Compute mean train log-likelihood per frame
      - Keep the n_states with best mean log-likelihood
      - Return final HMM and chosen n_states

    Returns:
      - models:       dict[user_id] -> GaussianHMM
      - best_states:  dict[user_id] -> chosen n_states
    """
    models: Dict[str, GaussianHMM] = {}
    best_states: Dict[str, int] = {}
    users = sorted(sequences.keys())

    for uid in users:
        train_seqs = sequences[uid]["train"]
        if not train_seqs:
            print(f"[WARN] No training sequences for user {uid}; skipping HMM.")
            continue

        best_model = None
        best_ll = -np.inf
        best_k = None

        for k in n_states_grid:
            try:
                model_k, mean_ll_k = _fit_hmm_for_user(
                    uid,
                    train_seqs,
                    n_states=k,
                    covariance_type=covariance_type,
                    n_iter=n_iter,
                )
                print(
                    f"[INFO] User {uid}: n_states={k} -> mean train loglik/frame={mean_ll_k:.4f}"
                )
                if mean_ll_k > best_ll:
                    best_ll = mean_ll_k
                    best_model = model_k
                    best_k = k
            except Exception as e:
                print(f"[WARN] User {uid}: failed to train HMM with n_states={k}: {e}")

        if best_model is None or best_k is None:
            print(f"[WARN] No valid HMM found for user {uid}; skipping this user.")
            continue

        models[uid] = best_model
        best_states[uid] = best_k
        print(
            f"[INFO] Selected HMM for user {uid}: n_states={best_k}, "
            f"mean train loglik/frame={best_ll:.4f}"
        )

    if not models:
        raise SystemExit("[FATAL] No HMMs trained successfully for any user.")

    return models, best_states


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_session_level(
    sequences: Dict[str, Dict[str, List[np.ndarray]]],
    models: Dict[str, GaussianHMM],
) -> Dict[str, float]:
    """
    Evaluate classification accuracy at SESSION level.

    For each test session (per user):
      - Compute log-likelihood under every user's HMM
      - Predict user = argmax likelihood
    """
    users = sorted(sequences.keys())

    y_true = []
    y_pred = []

    for uid_true in users:
        test_seqs = sequences[uid_true]["test"]
        if not test_seqs:
            print(f"[WARN] No test sessions for user {uid_true}; skipping.")
            continue

        for seq in test_seqs:
            seq = np.asarray(seq, dtype=float)
            scores = []
            for uid_model, model in models.items():
                try:
                    ll = model.score(seq)
                except Exception:
                    ll = -1e9  # if something blows up, treat as very bad
                scores.append((uid_model, ll))

            if not scores:
                continue

            # Predict user with max log-likelihood
            uid_hat = max(scores, key=lambda x: x[1])[0]
            y_true.append(uid_true)
            y_pred.append(uid_hat)

    y_true = np.array(y_true, dtype=str)
    y_pred = np.array(y_pred, dtype=str)

    if y_true.size == 0:
        raise SystemExit("[FATAL] No test sequences available for evaluation.")

    acc = float((y_true == y_pred).mean())
    print(
        f"[RESULT] Session-level HMM accuracy: {acc*100:.2f}% "
        f"over {len(y_true)} test sessions."
    )

    # Per-user mean accuracy (for report)
    per_user_acc: Dict[str, float] = {}
    for uid in sorted(set(y_true)):
        mask = (y_true == uid)
        per_user_acc[uid] = float((y_true[mask] == y_pred[mask]).mean())

    return {
        "overall_acc": acc,
        "n_test_sessions": float(len(y_true)),
        "per_user_mean_acc": float(np.mean(list(per_user_acc.values()))),
    }


# ---------------------------------------------------------------------
# Saving artifacts
# ---------------------------------------------------------------------

def save_models_and_metrics(
    models: Dict[str, GaussianHMM],
    scaler: StandardScaler,
    metrics: Dict[str, float],
    best_states: Dict[str, int],
) -> None:
    # Save scaler
    scaler_path = MODELS_DIR / "kb_hmm_scaler.joblib"
    dump(scaler, scaler_path)

    # Save HMM per user
    for uid, model in models.items():
        path = MODELS_DIR / f"hmm_{uid}.joblib"
        dump(model, path)

    # Save metrics as a small CSV for the report
    metrics_path = ARTIFACTS_DIR / "kb_hmm_metrics.csv"
    mean_n_states = float(np.mean(list(best_states.values()))) if best_states else 0.0

    df_metrics = pd.DataFrame(
        {
            "overall_acc": [metrics["overall_acc"]],
            "n_test_sessions": [metrics["n_test_sessions"]],
            "per_user_mean_acc": [metrics["per_user_mean_acc"]],
            "mean_n_states": [mean_n_states],
        }
    )
    df_metrics.to_csv(metrics_path, index=False)

    # Save chosen n_states per user
    best_states_path = ARTIFACTS_DIR / "kb_hmm_best_states.csv"
    df_states = pd.DataFrame(
        {
            "user_id": list(best_states.keys()),
            "n_states": list(best_states.values()),
        }
    ).sort_values("user_id")
    df_states.to_csv(best_states_path, index=False)

    print(f"[INFO] Saved HMM scaler -> {scaler_path}")
    print(f"[INFO] Saved per-user HMMs -> {MODELS_DIR}/hmm_*.joblib")
    print(f"[INFO] Saved metrics -> {metrics_path}")
    print(f"[INFO] Saved best n_states per user -> {best_states_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("[INFO] Loading keyboard CMU windows…")
    df = load_kb_df()

    print("[INFO] Building sequences and global scaler…")
    sequences, scaler = build_sequences(df)

    print("[INFO] Training Gaussian HMMs per user with n_states tuning…")
    models, best_states = train_hmms_with_tuning(
        sequences,
        n_states_grid=N_STATES_GRID,
        covariance_type="diag",
        n_iter=40,
    )

    print("[INFO] Evaluating session-level accuracy…")
    metrics = evaluate_session_level(sequences, models)

    print("[INFO] Saving models + metrics…")
    save_models_and_metrics(models, scaler, metrics, best_states)

    print("[DONE] Keyboard HMM training + evaluation complete.")


if __name__ == "__main__":
    main()
