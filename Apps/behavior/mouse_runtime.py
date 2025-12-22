# Apps/behavior/mouse_runtime.py
import numpy as np
from pathlib import Path
import joblib
import json

class MouseSVMEvaluator:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)

        self.scaler_path = self.model_dir / "mouse_scaler.joblib"
        self.model_path = self.model_dir / "mouse_svm.joblib"
        self.meta_path = self.model_dir / "mouse_meta.json"

        self.loaded = False
        self.feature_order = []
        self.users = []

        self._load()

    def _load(self):
        if (not self.scaler_path.exists()) or (not self.model_path.exists()):
            return

        self.scaler = joblib.load(self.scaler_path)
        self.model = joblib.load(self.model_path)

        if self.meta_path.exists():
            meta = json.load(open(self.meta_path))
            self.feature_order = meta.get("feature_order", [])
            self.users = meta.get("users", [])

        self.loaded = True

    def is_ready(self):
        return self.loaded

    def predict_user(self, feats: dict):
        """
        feats: dict of {feature_name: value}
        """
        if not self.loaded:
            return None

        vec = np.array([feats[k] for k in self.feature_order], dtype=float).reshape(1, -1)
        vec = self.scaler.transform(vec)

        probs = self.model.predict_proba(vec)[0]
        pred_idx = int(np.argmax(probs))
        pred_user = self.users[pred_idx]
        pred_prob = float(probs[pred_idx])

        return {
            "pred_user": pred_user,
            "prob": pred_prob,
            "probs_full": probs.tolist(),
        }
