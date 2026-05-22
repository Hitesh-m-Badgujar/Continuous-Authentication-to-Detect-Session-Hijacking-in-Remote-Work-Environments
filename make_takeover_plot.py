from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "Data" / "keyboard_test_windows.csv"
MODEL_DIR = BASE / "Models" / "kb_svm"
OUT_DIR = BASE / "artifacts" / "realtime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_MS = 150
SMOOTH_N = 10
ALLOW_T = 0.60
LOCK_T = 0.35

NON_FEATURE = {
    "user_id",
    "session_id",
    "window_id",
    "start_idx",
    "end_idx",
    "ks_count",
    "ks_unique",
    "file",
    "session",
}


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Stable softmax across rows."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def margin_trust(dec_row: np.ndarray) -> float:
    """
    Margin-based trust is kept only for comparison/debug.
    It is not used for the final takeover-aware policy.
    """
    top2 = np.partition(np.asarray(dec_row, dtype=np.float64), -2)[-2:]
    m = float(top2[1] - top2[0])
    return 1.0 / (1.0 + np.exp(-m))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create keyboard takeover replay plot from held-out test windows")
    p.add_argument("--user1", type=str, default=None, help="Enrolled user / claimed user")
    p.add_argument("--user2", type=str, default=None, help="Attacker user after takeover")
    p.add_argument("--n-each", type=int, default=120, help="Number of windows per user to replay")
    p.add_argument("--window-ms", type=int, default=WINDOW_MS, help="Synthetic replay spacing in milliseconds")
    p.add_argument("--smooth-n", type=int, default=SMOOTH_N, help="Rolling mean window")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not DATA.is_file():
        raise FileNotFoundError(f"Missing held-out keyboard test file: {DATA}")

    model = joblib.load(MODEL_DIR / "kb_svm_model.joblib")
    scaler = joblib.load(MODEL_DIR / "kb_svm_scaler.joblib")

    df = pd.read_csv(DATA)
    label_col = "user_id"
    if label_col not in df.columns:
        raise ValueError(f"{DATA} must contain '{label_col}'")

    users = sorted(df[label_col].astype(str).unique())
    if len(users) < 2:
        raise ValueError("Need at least 2 users in the held-out test split.")

    u1 = args.user1 if args.user1 is not None else ("s002" if "s002" in users else users[0])
    remaining = [u for u in users if u != u1]
    if not remaining:
        raise ValueError(f"Could not find a second user different from {u1!r}.")
    u2 = args.user2 if args.user2 is not None else ("s003" if "s003" in remaining else remaining[0])

    if u1 not in users:
        raise ValueError(f"Chosen user1 {u1!r} not found in {DATA.name}. Users: {users[:10]}...")
    if u2 not in users:
        raise ValueError(f"Chosen user2 {u2!r} not found in {DATA.name}. Users: {users[:10]}...")
    if u1 == u2:
        raise ValueError("user1 and user2 must be different.")

    feat_cols = [c for c in df.columns if c not in NON_FEATURE]
    expected = int(getattr(scaler, "n_features_in_", len(feat_cols)))
    if len(feat_cols) != expected:
        raise ValueError(
            f"Feature mismatch: got {len(feat_cols)} features, scaler expects {expected}.\n"
            f"Using file: {DATA}\n"
            f"Selected features: {feat_cols}\n"
            f"All columns: {list(df.columns)}"
        )

    df_u1 = df[df[label_col].astype(str) == u1]
    df_u2 = df[df[label_col].astype(str) == u2]
    if df_u1.empty or df_u2.empty:
        raise ValueError(f"No rows found for takeover users {u1!r} and {u2!r}")

    n1 = min(args.n_each, len(df_u1))
    n2 = min(args.n_each, len(df_u2))
    if n1 < 2 or n2 < 2:
        raise ValueError(
            f"Not enough held-out windows for replay. user1={u1} has {len(df_u1)}, user2={u2} has {len(df_u2)}"
        )

    a = df_u1.sample(n=n1, random_state=1, replace=False).reset_index(drop=True)
    b = df_u2.sample(n=n2, random_state=2, replace=False).reset_index(drop=True)

    session = pd.concat([a, b], ignore_index=True)
    true_user = [u1] * len(a) + [u2] * len(b)

    X = session[feat_cols].to_numpy(dtype=np.float32)
    X = scaler.transform(X)

    dec = model.decision_function(X)
    if dec.ndim != 2:
        raise RuntimeError(f"Expected multi-class decision scores with 2D shape, got {dec.shape}")

    classes = list(map(str, model.classes_))
    if u1 not in classes:
        raise ValueError(f"Enrolled user {u1!r} not found in model classes. Classes: {classes[:10]} ...")

    u1_idx = classes.index(u1)

    probs = softmax_rows(dec)
    trust = probs[:, u1_idx]  # takeover-aware trust of the enrolled/claimed user
    trust_margin = np.array([margin_trust(d) for d in dec], dtype=np.float64)
    trust_s = pd.Series(trust).rolling(args.smooth_n, min_periods=1).mean().to_numpy()

    action = np.where(
        trust_s < LOCK_T,
        "LOCK",
        np.where(trust_s < ALLOW_T, "STEP_UP", "ALLOW"),
    )

    t_ms = np.arange(len(trust_s)) * int(args.window_ms)
    takeover_t_ms = len(a) * int(args.window_ms)

    out = pd.DataFrame(
        {
            "t_ms": t_ms,
            "true_user": true_user,
            "claimed_user": u1,
            "attacker_user": u2,
            "trust_u1": trust,
            "trust_u1_smooth": trust_s,
            "trust_margin": trust_margin,
            "action": action,
            "source_csv": str(DATA),
        }
    )

    out_csv = OUT_DIR / "takeover_sim_kb.csv"
    out.to_csv(out_csv, index=False)

    plt.figure(figsize=(10, 4.8))
    plt.plot(t_ms, trust_s, label=f"trust(claimed user {u1}) smoothed")
    plt.axvline(takeover_t_ms, linestyle="--", label=f"takeover point ({u2})")
    plt.axhline(ALLOW_T, linestyle="--", label="ALLOW threshold")
    plt.axhline(LOCK_T, linestyle="--", label="LOCK threshold")
    plt.xlabel("time (ms)")
    plt.ylabel("trust")
    plt.title("Keyboard trust under simulated takeover (held-out test replay)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = OUT_DIR / "takeover_sim_kb.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Wrote:", out_csv)
    print("Wrote:", out_png)
    print(f"Users: u1(enrolled)={u1}, u2(attacker)={u2}")
    print(f"Feature count: {len(feat_cols)}")
    print(f"Held-out source: {DATA}")
    print(f"Min/Max trust_u1_smooth: {trust_s.min():.4f} / {trust_s.max():.4f}")


if __name__ == "__main__":
    main()
