# Apps/behavior/rt_scorer.py
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from joblib import load
from tensorflow import keras

MODELS_DIR = Path("Models/cae_kb")

DEFAULT_FEATURE_ORDER = [
    # exactly the engineered columns produced by ingest/cleaning and used in training
    "ks_count", "ks_unique",
    "dwell_mean", "dwell_std", "dwell_p10", "dwell_p50", "dwell_p90",
    "dd_mean", "dd_std", "dd_p10", "dd_p50", "dd_p90",
    "ud_mean", "ud_std", "ud_p10", "ud_p50", "ud_p90",
    "backspace_rate", "burst_mean", "idle_frac",
    # optional meta we ignore in the AE but may exist in CSVs; we’ll drop if present
    # "user_id","session_id","start_idx","end_idx"
]

MIN_KEYS_FOR_DECISION = 30      # require a little context before making a call
TRUST_ALLOW = 0.70
TRUST_STEPUP = 0.40

def _load_json(p: Path, default: Any) -> Any:
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

class RuntimeScorer:
    """
    Loads the trained conditional autoencoder + scaler + thresholds.
    Provides robust scoring for:
      - global mode (no user_id)
      - conditional mode (user_id given, if present in thresholds)
    Handles feature ordering, missing fields, and minimum window size.
    """

    def __init__(self):
        self.ready = False
        self.feature_order = _load_json(MODELS_DIR / "feature_order.json", DEFAULT_FEATURE_ORDER)

        # Load scaler + model
        scaler_path = MODELS_DIR / "scaler.joblib"
        model_path = MODELS_DIR / "model.h5"
        thr_path   = MODELS_DIR / "thresholds.json"

        if not scaler_path.exists() or not model_path.exists() or not thr_path.exists():
            # Not ready; caller should show a clear error
            return

        self.scaler = load(scaler_path)
        self.model = keras.models.load_model(model_path, compile=False)
        self.thresholds: Dict[str, float] = _load_json(thr_path, {})

        # Compute a robust global tau from per-user taus (median)
        taus = [float(v.get("best_tau", v) if isinstance(v, dict) else v)
                for v in self.thresholds.values() if v is not None]
        self.tau_global = float(np.median(taus)) * (1.10 if len(taus) else 1.0)  # small safety margin

        # Fallback if thresholds empty (shouldn’t happen after your calibration step)
        if not np.isfinite(self.tau_global) or self.tau_global <= 0:
            self.tau_global = 0.25  # typical DSL magnitude

        self.ready = True

    def _vectorize(self, feats: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        """Map dict -> ordered vector; fill missing with 0.  Also return ks_count for min-keys logic."""
        x = []
        for k in self.feature_order:
            v = feats.get(k, 0.0)
            try:
                x.append(float(v))
            except Exception:
                x.append(0.0)
        ks_count = int(round(feats.get("ks_count", 0)))
        return np.asarray(x, dtype=np.float32), ks_count

    @staticmethod
    def _trust_from_residual(residual: float, tau: float) -> float:
        """
        Smooth mapping:
        - residual == tau  -> ~0.5
        - residual << tau  -> -> 1
        - residual >> tau  -> -> 0
        """
        tau = max(1e-6, float(tau))
        # scale relative error and pass through logistic
        rel = (residual - tau) / (0.75 * tau)  # 0.75 sharpness is a good starting point for DSL
        t = 1.0 / (1.0 + np.exp(rel))
        # clamp to [0,1]
        return float(np.clip(t, 0.0, 1.0))

    @staticmethod
    def _action_from_trust(trust: float) -> str:
        if trust >= TRUST_ALLOW:
            return "ALLOW"
        if trust >= TRUST_STEPUP:
            return "STEP_UP"
        return "LOCK"

    def _score_vec(self, vec: np.ndarray, tau: float) -> Dict[str, Any]:
        # Standardize then reconstruct
        z = self.scaler.transform(vec.reshape(1, -1))
        recon = self.model.predict(z, verbose=0)
        # Reconstruction domain: model was trained on standardized features
        # Residual as mean squared error per sample
        resid = float(np.mean((z - recon) ** 2))
        trust = self._trust_from_residual(resid, tau)
        return {
            "residual": resid,
            "tau": float(tau),
            "trust_instant": trust,
            "action": self._action_from_trust(trust),
        }

    def score(self, features: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
        if not self.ready:
            return {"ok": False, "error": "model_not_loaded"}

        vec, ks_count = self._vectorize(features)

        # Enforce minimum window size so we don't decide on 3–5 keystrokes
        if ks_count < MIN_KEYS_FOR_DECISION:
            return {
                "ok": True,
                "mode": "warmup",
                "needed": max(0, MIN_KEYS_FOR_DECISION - ks_count),
                "trust_instant": 0.5,          # neutral
                "residual": None,
                "tau": None,
                "action": "WARN"
            }

        # Conditional if we have a per-user threshold; else global
        mode = "global"
        tau = self.tau_global
        if user_id:
            # normalize user id like in training (DSL users are like s0xx)
            key = str(user_id).strip()
            if key in self.thresholds:
                v = self.thresholds[key]
                tau = float(v.get("best_tau", v) if isinstance(v, dict) else v)
                mode = "conditional"

        out = self._score_vec(vec, tau)
        out.update({"ok": True, "mode": mode})
        return out
