# Apps/behavior/runtime_global.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import joblib
from keras.models import load_model

# Reuse same feature ordering used by keyboard windows
from Apps.behavior.runtime_kb import FEATURES as KB_FEATURES

MODELS_DIR = Path("Models/ae_global")

@dataclass
class GlobalRuntime:
    scaler: Any
    model: Any
    tau: float  # global threshold

    def vectorize(self, feat_dict: Dict[str, Any]) -> np.ndarray:
        x = np.array([float(feat_dict.get(k, 0.0)) for k in KB_FEATURES], dtype=np.float32)
        return x

    def residual(self, x_raw: np.ndarray) -> float:
        x = self.scaler.transform(x_raw[None, :])
        xr = self.model.predict(x, verbose=0)
        return float(np.mean((x - xr) ** 2))

    def score(self, feat_dict: Dict[str, Any]) -> Dict[str, Any]:
        x = self.vectorize(feat_dict)
        r = self.residual(x)
        tau = self.tau

        # trust: 1 at r=0, ~0 at r >= 2*tau
        trust = max(0.0, 1.0 - (r / (2.0 * tau + 1e-12)))

        if trust >= 0.70:
            action = "ALLOW"
        elif trust >= 0.50:
            action = "STEP_UP"
        elif trust >= 0.35:
            action = "WARN"
        else:
            action = "LOCK"

        return {"ok": True, "residual": r, "tau": tau, "trust": trust, "action": action}

_runtime_cache: GlobalRuntime | None = None

def get_global_scorer() -> GlobalRuntime:
    global _runtime_cache
    if _runtime_cache is not None:
        return _runtime_cache

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    model = load_model(MODELS_DIR / "model.h5", compile=False)

    tau = None
    tau_path = MODELS_DIR / "tau.json"
    if tau_path.exists():
        tau = float(json.loads(tau_path.read_text())["tau"])
    else:
        raise RuntimeError("Models/ae_global/tau.json missing. Train the global AE.")

    _runtime_cache = GlobalRuntime(scaler=scaler, model=model, tau=tau)
    return _runtime_cache
