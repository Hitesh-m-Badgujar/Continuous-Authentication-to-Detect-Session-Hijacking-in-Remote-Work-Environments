import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent  # H1
DATA = BASE / "Data" / "kb_cmu_windows.csv"
MODEL_DIR = BASE / "Models" / "kb_svm"
OUT_DIR = BASE / "artifacts" / "realtime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = joblib.load(MODEL_DIR / "kb_svm_model.joblib")
SCALER = joblib.load(MODEL_DIR / "kb_svm_scaler.joblib")

WINDOW_MS = 150
SMOOTH_N = 10
ALLOW_T = 0.60
LOCK_T = 0.35


def margin_trust(dec_row: np.ndarray) -> float:
    """
    Margin-based trust (top1-top2) is NOT takeover-aware.
    It stays high if the classifier is confident about *any* class.
    Kept here only for comparison/debug.
    """
    top2 = np.partition(dec_row, -2)[-2:]
    m = float(top2[1] - top2[0])
    return 1.0 / (1.0 + np.exp(-m))


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """
    Stable softmax across rows.
    Used to convert SVC decision_function scores to pseudo-probabilities.
    """
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


df = pd.read_csv(DATA)

label_col = "user_id"  # this file uses user_id
NON_FEATURE = {
    label_col,
    "session_id",
    "window_id",
    "start_idx",
    "end_idx",
    "ks_count",
    "ks_unique",
    "file",
    "session",
}

expected = int(getattr(SCALER, "n_features_in_", 18))
feat_cols = [c for c in df.columns if c not in NON_FEATURE]

# Hard check so you don't waste time later
if len(feat_cols) != expected:
    raise ValueError(
        f"Feature mismatch: got {len(feat_cols)} features, scaler expects {expected}.\n"
        f"Using file: {DATA}\n"
        f"Selected features: {feat_cols}\n"
        f"All columns: {list(df.columns)}"
    )

# pick two users
users = df[label_col].unique()
if len(users) < 2:
    raise ValueError("Need at least 2 users in dataset.")

u1, u2 = users[0], users[1]

# take enough windows per user
a = df[df[label_col] == u1].sample(120, random_state=1).reset_index(drop=True)
b = df[df[label_col] == u2].sample(120, random_state=2).reset_index(drop=True)

session = pd.concat([a, b], ignore_index=True)
true_user = [u1] * len(a) + [u2] * len(b)

X = session[feat_cols].values
X = SCALER.transform(X)

# Multi-class decision scores: (n_samples, n_classes)
dec = MODEL.decision_function(X)

# --- FIX: takeover-aware trust = probability of the ENROLLED/CLAIMED user (u1)
classes = list(MODEL.classes_)
if u1 not in classes:
    raise ValueError(f"Enrolled user {u1} not found in model classes. Classes: {classes[:10]} ...")

u1_idx = classes.index(u1)

probs = softmax_rows(dec)          # pseudo-probabilities from decision scores
trust = probs[:, u1_idx]           # <-- the key change: trust of the claimed user

# Optional: margin trust for comparison (not used for action/policy)
trust_margin = np.array([margin_trust(d) for d in dec])

# Smooth trust
trust_s = pd.Series(trust).rolling(SMOOTH_N, min_periods=1).mean().values

# Policy from smoothed takeover-aware trust
action = np.where(
    trust_s < LOCK_T,
    "LOCK",
    np.where(trust_s < ALLOW_T, "STEP_UP", "ALLOW")
)

t_ms = np.arange(len(trust_s)) * WINDOW_MS

out = pd.DataFrame({
    "t_ms": t_ms,
    "true_user": true_user,
    "trust_u1": trust,
    "trust_u1_smooth": trust_s,
    "trust_margin": trust_margin,
    "action": action
})

out_csv = OUT_DIR / "takeover_sim_kb.csv"
out.to_csv(out_csv, index=False)

# Plot takeover-aware trust
plt.figure()
plt.plot(t_ms, trust_s, label="trust(enrolled user u1) smoothed")
plt.axvline(len(a) * WINDOW_MS, linestyle="--", label="takeover point")
plt.axhline(ALLOW_T, linestyle="--", label="ALLOW threshold")
plt.axhline(LOCK_T, linestyle="--", label="LOCK threshold")
plt.xlabel("time (ms)")
plt.ylabel("trust")
plt.title("Keyboard trust under simulated takeover (dataset replay)")
plt.legend()
plt.tight_layout()

out_png = OUT_DIR / "takeover_sim_kb.png"
plt.savefig(out_png, dpi=200)

print("Wrote:", out_csv)
print("Wrote:", out_png)
print(f"Users: u1(enrolled)={u1}, u2(attacker)={u2}")
print(f"Feature count: {len(feat_cols)}")
print(f"Min/Max trust_u1_smooth: {trust_s.min():.4f} / {trust_s.max():.4f}")
