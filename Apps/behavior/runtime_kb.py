from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from Apps.behavior.ae_conditional import load_runtime, residuals

MODELS_DIR = Path("Models/cae_kb")

# Keep in sync with training feature order
FEATURES = [
    "dwell_mean","dwell_std","dwell_p10","dwell_p50","dwell_p90",
    "dd_mean","dd_std","dd_p10","dd_p50","dd_p90",
    "ud_mean","ud_std","ud_p10","ud_p50","ud_p90",
    "backspace_rate","burst_mean","idle_frac"
]

class KBScorer:
    def __init__(self):
        scaler, model, feature_names, th_json, user_to_idx = load_runtime(MODELS_DIR)
        # tolerate missing 'values' during early runs
        thresholds = th_json.get("values", {})
        self.scaler = scaler
        self.model = model
        self.user_to_idx = user_to_idx
        self.thresholds = thresholds
        self.features = FEATURES

    def has_user(self, user_id: str) -> bool:
        return str(user_id) in self.user_to_idx and str(user_id) in self.thresholds

    def vectorize(self, feat_dict: dict) -> np.ndarray:
        """Map incoming dict to training order; raise if any missing."""
        missing = [f for f in self.features if f not in feat_dict]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        x = np.array([feat_dict[f] for f in self.features], dtype=np.float32)
        if not np.all(np.isfinite(x)):
            raise ValueError("Non-finite values in features")
        return x[None, :]  # shape (1, D)

    def score(self, user_id: str, feat_dict: dict) -> dict:
        """Return residual, threshold and a 0–1 trust score."""
        uid = str(user_id)
        if not self.has_user(uid):
            return {"ok": False, "error": f"user '{uid}' unknown to model"}
        X = self.vectorize(feat_dict)
        uid_arr = np.array([self.user_to_idx[uid]], dtype=np.int32)
        r = float(residuals(self.model, self.scaler, X, uid_arr)[0])
        tau = float(self.thresholds[uid])

        # Turn residual into trust (higher = better). Beta controls slope.
        beta = 0.10 * tau if tau > 0 else 0.05
        z = (r - tau) / max(beta, 1e-6)
        # squashed to [0,1]
        trust = 1.0 / (1.0 + np.exp(z))

        # policy suggestion
        if r <= 0.7 * tau:
            action = "ALLOW"
        elif r <= tau:
            action = "STEP_UP"   # e.g., ask for face or token
        elif r <= 1.5 * tau:
            action = "WARN"      # soft lock if this persists
        else:
            action = "LOCK"

        return {"ok": True, "residual": r, "tau": tau, "trust": float(trust), "action": action}

# Singleton for reuse
_kb_scorer = None
def get_kb_scorer() -> KBScorer:
    global _kb_scorer
    if _kb_scorer is None:
        _kb_scorer = KBScorer()
    return _kb_scorer
