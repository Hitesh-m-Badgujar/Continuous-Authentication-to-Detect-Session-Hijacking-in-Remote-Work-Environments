from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
from joblib import load as joblib_load
from sklearn.base import ClassifierMixin


# Paths / constants
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "Models"

# 18-D keyboard feature schema (keep exactly this order)
FEATURE_COLS: List[str] = [
    "dwell_mean", "dwell_std", "dwell_p10", "dwell_p50", "dwell_p90",
    "dd_mean", "dd_std", "dd_p10", "dd_p50", "dd_p90",
    "ud_mean", "ud_std", "ud_p10", "ud_p50", "ud_p90",
    "backspace_rate", "burst_mean", "idle_frac",
]

DEFAULT_MODEL_DIR = MODELS_DIR / "kb_svm"   # expect: scaler.joblib, kb_svm.joblib


def _clip01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v):
        return 0.0
    return max(0.0, min(1.0, v))


@dataclass
class KeyboardSVMScorer:
    """
    Keyboard SVM scorer (no user ID).
    - Uses predict_proba if available (prefers proba[:,1] as "genuine" prob)
    - Else applies logistic to decision_function: trust = 1/(1+exp(-s/scale))
    """
    model_dir: Path
    scaler: any
    model: ClassifierMixin
    decision_scale: float = 2.0  # soften if your margins are large

    def __init__(self, model_dir: Optional[str | Path] = None) -> None:
        md = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        scaler_p = md / "scaler.joblib"
        model_p = md / "kb_svm.joblib"

        if not scaler_p.is_file():
            raise FileNotFoundError(f"Keyboard scaler not found: {scaler_p}")
        if not model_p.is_file():
            raise FileNotFoundError(f"Keyboard SVM not found: {model_p}")

        self.model_dir = md
        self.scaler = joblib_load(scaler_p)
        self.model = joblib_load(model_p)

    def health(self) -> Dict[str, object]:
        # Try to detect if model supports predict_proba
        has_proba = hasattr(self.model, "predict_proba")
        return {
            "ok": True,
            "dim": len(FEATURE_COLS),
            "has_predict_proba": bool(has_proba),
        }

    def _trust_from_scores(self, x_scaled: np.ndarray) -> Tuple[float, float]:
        """
        Returns (trust, raw_score). Prefers predict_proba if available.
        """
        m = self.model
        # Case 1: true probabilities
        if hasattr(m, "predict_proba"):
            proba = m.predict_proba(x_scaled)[0]  # shape (n_classes,)
            # If binary classifier with classes [0,1], use P(class=1)
            if hasattr(m, "classes_") and len(m.classes_) == 2:
                idx1 = int(np.where(m.classes_ == 1)[0][0]) if 1 in m.classes_ else int(np.argmax(proba))
                trust = float(proba[idx1])
            else:
                # Multiclass: confidence = max prob; convert to trust
                trust = float(np.max(proba))
            raw = float(np.max(proba))
            return _clip01(trust), raw

        # Case 2: decision_function → logistic
        if hasattr(m, "decision_function"):
            s = float(m.decision_function(x_scaled)[0])
            trust = 1.0 / (1.0 + math.exp(-s / max(self.decision_scale, 1e-6)))
            return _clip01(trust), s

        # Fallback: predict → {0,1}
        yhat = int(m.predict(x_scaled)[0])
        return (1.0 if yhat == 1 else 0.0), float(yhat)

    def score(self, feats: np.ndarray) -> Dict[str, object]:
        X = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        if X.shape[1] != len(FEATURE_COLS):
            raise ValueError(f"Keyboard feature dim mismatch: got {X.shape[1]}, expected {len(FEATURE_COLS)}")
        Xs = self.scaler.transform(X)
        trust, raw = self._trust_from_scores(Xs)
        action = "ALLOW" if trust >= 0.75 else ("STEP_UP" if trust >= 0.40 else "LOCK")
        return {
            "ok": True,
            "mode": "svm",
            "trust": float(trust),
            "raw": float(raw),
            "action": action,
        }
