# Apps/behavior/ae_conditional.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import math
import numpy as np
from joblib import load as joblib_load

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"
MODELS_DIR = BASE_DIR / "Models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Keyboard window features (18-D)
FEATURE_COLS: List[str] = [
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

# Default location for keyboard SVM artifacts
# (You must have Models/kb_svm/kb_svm_scaler.joblib and kb_svm_model.joblib)
DEFAULT_MODEL_DIR = MODELS_DIR / "kb_svm"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_model_and_scaler(model_dir: Path) -> Tuple[Any, Any]:
    """
    Load keyboard SVM scaler and model from model_dir.

    Expected filenames:
      - kb_svm_scaler.joblib
      - kb_svm_model.joblib
    """
    model_dir = Path(model_dir)
    scaler_path = model_dir / "kb_svm_scaler.joblib"
    model_path = model_dir / "kb_svm_model.joblib"

    if not scaler_path.is_file():
        raise FileNotFoundError(f"Keyboard kb_svm_scaler.joblib not found at {scaler_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Keyboard kb_svm_model.joblib not found at {model_path}")

    scaler = joblib_load(scaler_path)
    model = joblib_load(model_path)

    # Sanity check on input dimension if available
    try:
        n_features = getattr(scaler, "n_features_in_", None)
        if n_features is not None and n_features != len(FEATURE_COLS):
            print(
                f"[WARN] Keyboard SVM expects {n_features} features, "
                f"FEATURE_COLS has {len(FEATURE_COLS)}."
            )
    except Exception:
        pass

    return scaler, model


def _clip01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v):
        return 0.0
    return max(0.0, min(1.0, v))


def _policy(trust: float) -> str:
    """
    Simple keyboard-only policy:

      trust >= 0.75 -> ALLOW
      0.40–0.75     -> STEP_UP
      < 0.40        -> LOCK
    """
    if trust >= 0.75:
        return "ALLOW"
    if trust >= 0.40:
        return "STEP_UP"
    return "LOCK"


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim == 0:
        logits = logits.reshape(1)
    m = np.max(logits)
    exps = np.exp(logits - m)
    s = np.sum(exps)
    if s <= 0:
        return np.ones_like(exps) / len(exps)
    return exps / s


def _margin_to_trust(margin: float, temperature: float = 1.0) -> float:
    """Map a top-1 vs top-2 margin to a [0,1] trust score.

    We avoid using raw softmax(max_score) as trust because SVC decision scores
    can be large and lead to visually "stuck" values in the live demo.
    """
    try:
        m = float(margin)
    except Exception:
        return 0.0
    if not np.isfinite(m) or m <= 0:
        return 0.0

    # Logistic squashing around 0 with a simple rescale to [0,1]
    t = max(1e-6, float(temperature))
    s = 1.0 / (1.0 + math.exp(-(m / t)))  # in (0.5, 1)
    trust = (s - 0.5) * 2.0              # in (0, 1)
    return _clip01(trust)


# ---------------------------------------------------------------------
# Runtime scorer (SVM)
# ---------------------------------------------------------------------

@dataclass
class RuntimeScorer:
    """
    Runtime scorer for keyboard SVM.

    NO per-user runtime. We treat SVM output as:
      - get decision scores from decision_function (or predict_proba if exists)
      - softmax to get pseudo-probabilities per class
      - trust = max softmax(prob) in [0,1]
    """

    model_dir: Path

    def __init__(self, model_dir: Optional[Path | str] = None) -> None:
        if model_dir is None:
            model_dir = DEFAULT_MODEL_DIR

        # If settings points to Models/cae_kb, transparently map to kb_svm
        p = Path(model_dir)
        if p.name == "cae_kb":
            p = p.parent / "kb_svm"
        self.model_dir = p

        self.scaler, self.model = _load_model_and_scaler(self.model_dir)

    # -------- health --------

    def health(self) -> Dict[str, Any]:
        try:
            n_features = getattr(self.scaler, "n_features_in_", None)
        except Exception:
            n_features = None
        try:
            classes = list(map(str, getattr(self.model, "classes_", [])))
        except Exception:
            classes = []

        return {
            "ok": True,
            "dim": int(n_features) if n_features is not None else len(FEATURE_COLS),
            "n_classes": len(classes),
            "classes": classes,
        }

    # -------- internal helpers --------

    def _proba_from_model(self, feats_row: np.ndarray) -> Tuple[np.ndarray, int, float, float]:
        """
        Compute "probabilities" for a single feature row.

        If model has predict_proba -> use it.
        Otherwise:
          - use decision_function
          - for binary: sigmoid(score) -> [p0, p1]
          - for multi-class: softmax(decision scores)
        """
        x = np.asarray(feats_row, dtype=np.float32).reshape(1, -1)

        # Align dimension with scaler expectations
        n_expected = getattr(self.scaler, "n_features_in_", None)
        if n_expected is not None and x.shape[1] != n_expected:
            if x.shape[1] < n_expected:
                pad = np.zeros((1, n_expected - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, pad], axis=1)
            else:
                x = x[:, :n_expected]

        x_scaled = self.scaler.transform(x)

        # Case 1: predict_proba available
        margin = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x_scaled)[0]
            proba = np.asarray(proba, dtype=float)
            if proba.size >= 2:
                top2 = np.partition(proba, -2)[-2:]
                margin = float(np.max(top2) - np.min(top2))
        else:
            # Case 2: use decision_function
            df = self.model.decision_function(x_scaled)
            df = np.asarray(df, dtype=float)
            if df.ndim == 1:
                # One score -> binary
                s = df[0]
                p1 = 1.0 / (1.0 + math.exp(-float(s)))
                p0 = 1.0 - p1
                proba = np.array([p0, p1], dtype=float)
                margin = float(abs(s))
            elif df.ndim == 2 and df.shape[1] >= 2:
                logits = df[0]
                # softmax is used only for display/probability-like output
                proba = _softmax(logits)
                top2 = np.partition(logits, -2)[-2:]
                margin = float(np.max(top2) - np.min(top2))
            else:
                n = df.shape[-1] if df.ndim > 1 else 2
                proba = np.ones(int(n), dtype=float) / float(n)
                margin = 0.0

        idx = int(np.argmax(proba))
        max_prob = float(proba[idx])
        return proba, idx, max_prob, margin

    # -------- scoring API --------

    def score_global(self, feats: np.ndarray) -> Dict[str, Any]:
        """
        Score keyboard features in 'global' mode.

        feats: array-like, shape (1, n_features) or (n_features,)
        """
        x = np.asarray(feats, dtype=np.float32).reshape(1, -1)
        proba, idx, max_prob, margin = self._proba_from_model(x[0])
        classes = list(map(str, getattr(self.model, "classes_", [])))
        pred_user = classes[idx] if classes and idx < len(classes) else None

        # Margin-based trust is more sensitive in the live demo.
        trust = _margin_to_trust(margin, temperature=1.0)
        action = _policy(trust)

        return {
            "mode": "global",
            "trust": float(trust),
            "action": action,
            "prob": float(max_prob),
            "margin": float(margin),
            "pred_user": pred_user,
            "proba": {cls: float(p) for cls, p in zip(classes, proba)},
        }

    def score(
        self,
        feats: np.ndarray,
        user_id: Optional[str] = None,
        mode: str = "global",
    ) -> Dict[str, Any]:
        """
        Generic .score() used by views._kb_score_any. Ignores user_id/mode.
        """
        out = self.score_global(feats)
        out["user_id"] = None
        return out


# ---------------------------------------------------------------------
# Backwards-compat helper functions
# ---------------------------------------------------------------------

def load_runtime(model_dir: Optional[Path | str] = None) -> RuntimeScorer:
    return RuntimeScorer(model_dir=model_dir)


def residuals(X: np.ndarray, model_dir: Optional[Path | str] = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return np.zeros(X.shape[0], dtype=np.float32)
