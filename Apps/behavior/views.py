# Apps/behavior/views.py

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from joblib import load as joblib_load

from . import ae_conditional, trust_fusion, face_runtime
from .trust_logging import append_trust_row

log = logging.getLogger(__name__)

# Rolling history of fused trust per user_id
_SESSION_TRUST_HISTORY: Dict[str, list[float]] = {}

# ---------------------------------------------------------------------
# Keyboard scorer – RuntimeScorer singleton (SVM-based)
# ---------------------------------------------------------------------

_KB_SCORER: Optional[Any] = None


def _get_kb_scorer():
    """
    Lazy-load the keyboard RuntimeScorer from ae_conditional.
    """
    global _KB_SCORER
    if _KB_SCORER is not None:
        return _KB_SCORER

    # Keyboard runtime is SVM-based (Models/kb_svm). Some older parts of the
    # repo refer to "cae_kb"; keep the runtime pointing at the real SVM dir.
    model_dir = getattr(settings, "MODELS_DIR") / "kb_svm"
    scorer_cls = getattr(ae_conditional, "RuntimeScorer")
    _KB_SCORER = scorer_cls(model_dir=model_dir)
    return _KB_SCORER


def _kb_score_any(scorer: Any, feats: np.ndarray) -> Dict[str, Any]:
    """
    Call into RuntimeScorer without tying to one method name.
    Expects a dict with at least 'trust' and 'action'.
    """
    feats = np.asarray(feats, dtype="float32").reshape(1, -1)

    if hasattr(scorer, "score_global"):
        return scorer.score_global(feats)
    if hasattr(scorer, "score"):
        return scorer.score(feats)

    raise RuntimeError(
        "RuntimeScorer has no recognised scoring method (expected .score_global or .score)."
    )


# ---------------------------------------------------------------------
# Mouse scorer (SVM) – lazy singleton
# ---------------------------------------------------------------------

_MOUSE_MODEL = None
_MOUSE_SCALER = None
_MOUSE_META: Optional[Dict[str, Any]] = None


def _get_mouse_model():
    """
    Load mouse SVM + scaler trained by train_mouse_svm.py.

    Files:
      Models/mouse/mouse_scaler.joblib
      Models/mouse/mouse_svm.joblib
      Models/mouse/mouse_meta.json
    """
    global _MOUSE_MODEL, _MOUSE_SCALER, _MOUSE_META
    if _MOUSE_MODEL is not None and _MOUSE_SCALER is not None:
        return _MOUSE_MODEL, _MOUSE_SCALER, _MOUSE_META

    models_dir = getattr(settings, "MODELS_DIR") / "mouse"
    scaler_path = models_dir / "mouse_scaler.joblib"
    model_path = models_dir / "mouse_svm.joblib"
    meta_path = models_dir / "mouse_meta.json"

    try:
        _MOUSE_SCALER = joblib_load(scaler_path)
        _MOUSE_MODEL = joblib_load(model_path)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                _MOUSE_META = json.load(f)
        except FileNotFoundError:
            _MOUSE_META = None
    except FileNotFoundError:
        _MOUSE_MODEL = None
        _MOUSE_SCALER = None
        _MOUSE_META = None

    return _MOUSE_MODEL, _MOUSE_SCALER, _MOUSE_META


# ---------------------------------------------------------------------
# Mouse scoring helpers
# ---------------------------------------------------------------------

def _mouse_health_payload() -> Dict[str, Any]:
    model, scaler, meta = _get_mouse_model()
    ok = model is not None and scaler is not None
    payload: Dict[str, Any] = {
        "ok": ok,
        "has_model": model is not None,
        "has_scaler": scaler is not None,
    }
    if scaler is not None:
        try:
            n_features = getattr(scaler, "n_features_in_", None)
        except Exception:
            n_features = None
        payload["n_features"] = int(n_features) if n_features is not None else None

    if meta:
        payload["meta"] = meta

    return payload


def _mouse_score_any(features: Any) -> Dict[str, Any]:
    """
    Run mouse SVM on a single feature window.

    Returns:
      {
        "prob":  float,   # max class probability (raw SVM output)
        "trust": float,   # calibrated trust in [0,1]
        "proba": { class_id: prob, ... }
      }
    """
    model, scaler, meta = _get_mouse_model()
    if model is None or scaler is None:
        raise RuntimeError("mouse_model_not_loaded")

    x = np.asarray(features, dtype="float32").reshape(1, -1)
    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None and x.shape[1] != expected:
        raise ValueError(f"mouse feature dim mismatch: got {x.shape[1]}, expected {expected}")

    x_scaled = scaler.transform(x)
    proba = model.predict_proba(x_scaled)[0]
    classes = list(map(str, getattr(model, "classes_", [])))
    if not classes:
        raise RuntimeError("mouse_model_no_classes")

    idx = int(np.argmax(proba))
    pred_prob = float(proba[idx])
    n_classes = len(classes)

    trust_val = proba_to_trust(pred_prob, n_classes=n_classes)

    # We treat this calibrated trust as the mouse trust.
    return {
        "prob": pred_prob,
        "trust": float(trust_val),
        "proba": {cls: float(p) for cls, p in zip(classes, proba)},
    }


def proba_to_trust(pred_prob: float, n_classes: Optional[int] = None) -> float:
    """
    Map SVM max probability -> trust in [0,1].

    - Accounts for uniform baseline (1 / n_classes).
    - Applies a gamma < 1 to avoid everything being stuck near 0.3.
    """
    try:
        p = float(pred_prob)
    except Exception:
        return 0.0

    if not np.isfinite(p) or p <= 0.0:
        return 0.0

    # Remove uniform baseline (random guess)
    if n_classes and n_classes > 1:
        uniform = 1.0 / float(n_classes)
        if p <= uniform:
            t = 0.0
        else:
            t = (p - uniform) / (1.0 - uniform)
    else:
        t = p

    # Clamp
    t = max(0.0, min(1.0, t))

    # Gamma < 1 → boost mid-values a bit (e.g. 0.3 -> ~0.6)
    gamma = 0.3
    t = t ** gamma

    return float(max(0.0, min(1.0, t)))


# ---------------------------------------------------------------------
# Face engine singleton
# ---------------------------------------------------------------------

_FACE_ENGINE: Optional[face_runtime.FaceEngine] = None


def _get_face_engine() -> face_runtime.FaceEngine:
    global _FACE_ENGINE
    if _FACE_ENGINE is None:
        _FACE_ENGINE = face_runtime.FaceEngine()
    return _FACE_ENGINE


# ---------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------

def index(request: HttpRequest) -> HttpResponse:
    return redirect("behavior:monitor_page")


def monitor_page(request: HttpRequest) -> HttpResponse:
    # Keyboard health
    try:
        scorer = _get_kb_scorer()
        if hasattr(scorer, "health"):
            kb_info = scorer.health()
            if not isinstance(kb_info, dict):
                kb_info = {"ok": False, "error": "health_not_dict"}
        else:
            # Basic info only: feature dim + (optional) classes / internal_acc
            kb_info = {
                "ok": True,
                "dim": len(ae_conditional.FEATURE_COLS),
            }
    except Exception as e:
        log.exception("kb_health failed")
        kb_info = {"ok": False, "error": str(e)}

    # Mouse health
    try:
        mouse_info = _mouse_health_payload()
    except Exception as e:
        log.exception("mouse_health failed")
        mouse_info = {"ok": False, "error": str(e)}

    context = {
        "kb_health": json.dumps(kb_info, default=str),
        "mouse_health": json.dumps(mouse_info, default=str),
    }
    return render(request, "behavior/monitor.html", context)


def kb_health(request: HttpRequest) -> JsonResponse:
    try:
        scorer = _get_kb_scorer()
        if hasattr(scorer, "health"):
            payload = scorer.health()
            if not isinstance(payload, dict):
                payload = {"ok": False, "error": "health_not_dict"}
        else:
            payload = {
                "ok": True,
                "dim": len(ae_conditional.FEATURE_COLS),
            }
    except Exception as e:
        log.exception("kb_health error")
        payload = {"ok": False, "error": str(e)}

    return JsonResponse(payload)


def mouse_health(request: HttpRequest) -> JsonResponse:
    try:
        payload = _mouse_health_payload()
    except Exception as e:
        log.exception("mouse_health error")
        payload = {"ok": False, "error": str(e)}

    return JsonResponse(payload)


@csrf_exempt
def stream_keystrokes(request: HttpRequest) -> JsonResponse:
    """
    Combined endpoint for keyboard + mouse streaming.
    """
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception as e:
        return JsonResponse(
            {"ok": False, "error": "bad_json", "detail": str(e)}, status=400
        )

    kb_payload: Dict[str, Any] = {"ok": False, "skipped": True}
    mouse_payload: Dict[str, Any] = {"ok": False, "skipped": True}

    # Keyboard
    kb_feats = body.get("kb_features") or body.get("features")
    if kb_feats is not None:
        try:
            scorer = _get_kb_scorer()
            kb_res = _kb_score_any(scorer, kb_feats)
            kb_payload = {"ok": True, **kb_res}
        except Exception as e:
            log.exception("keyboard inference failed")
            kb_payload = {
                "ok": False,
                "error": "inference_failed",
                "detail": str(e),
            }

    # Mouse
    mouse_feats = (
        body.get("mouse_features")
        or body.get("mouse_window")
        or body.get("mouse")
    )
    if mouse_feats is not None:
        try:
            mouse_res = _mouse_score_any(mouse_feats)
            mt = float(mouse_res.get("trust", 0.0))
            mouse_payload = {
                "ok": True,
                "mouse_trust": mt,
                "prob": float(mouse_res.get("prob", 0.0)),
            }
        except Exception as e:
            log.exception("mouse inference failed")
            mouse_payload = {
                "ok": False,
                "error": "mouse_inference_failed",
                "detail": str(e),
            }

    if kb_feats is None and mouse_feats is None:
        return JsonResponse(
            {
                "ok": False,
                "error": "no_features",
                "detail": "Provide kb_features or mouse_features.",
            },
            status=400,
        )

    return JsonResponse({"ok": True, "keyboard": kb_payload, "mouse": mouse_payload})


@csrf_exempt
def stream_mouse(request: HttpRequest) -> JsonResponse:
    """
    Mouse-only streaming endpoint (if you ever want to test it separately).
    """
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST required"}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception as e:
        return JsonResponse(
            {"ok": False, "error": "bad_json", "detail": str(e)}, status=400
        )

    feats = body.get("features") or body.get("mouse_features") or body.get("mouse")
    if feats is None:
        return JsonResponse(
            {
                "ok": False,
                "error": "no_mouse_features",
                "detail": "Expected 'features' list.",
            },
            status=400,
        )

    try:
        res = _mouse_score_any(feats)
        mt = float(res.get("trust", 0.0))
        return JsonResponse(
            {
                "ok": True,
                "mouse_trust": mt,
                "prob": float(res.get("prob", 0.0)),
            }
        )
    except Exception as e:
        log.exception("mouse-only inference failed")
        return JsonResponse(
            {"ok": False, "error": "mouse_inference_failed", "detail": str(e)},
            status=500,
        )


def _to_float_or_none(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


@csrf_exempt
def fuse_scores(request: HttpRequest) -> JsonResponse:
    """
    Fuse kb_trust + mouse_trust + face_match + liveness
    into behaviour_trust, face_trust, overall_trust and policy.
    """
    if request.method != "POST":
        return JsonResponse(
            {"ok": False, "error": "method_not_allowed", "detail": "POST required"},
            status=405,
        )

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception as e:
        return JsonResponse(
            {"ok": False, "error": "bad_json", "detail": str(e)}, status=400
        )

    # In this demo we use a single global session/user.
    user_id = "global"

    kb_trust = _to_float_or_none(body.get("kb_trust"))
    mouse_trust = _to_float_or_none(body.get("mouse_trust"))
    face_match = _to_float_or_none(body.get("face_match"))
    liveness = _to_float_or_none(body.get("liveness"))

    behaviour_trust = trust_fusion.fuse_behaviour(kb_trust, mouse_trust)
    face_trust = trust_fusion.fuse_face(face_match, liveness)
    overall_trust = trust_fusion.fuse_overall(behaviour_trust, face_trust)

    if overall_trust is not None:
        hist = _SESSION_TRUST_HISTORY.setdefault(user_id, [])
        hist.append(float(overall_trust))
        if len(hist) > 100:
            hist.pop(0)
        overall_trust_rolling: Optional[float] = float(np.mean(hist))
    else:
        overall_trust_rolling = None

    action = trust_fusion.trust_policy_action(overall_trust)

    # --------- NEW: log to live_trust_timeseries.csv ----------
    try:
        fused_for_log = overall_trust_rolling if overall_trust_rolling is not None else overall_trust

        append_trust_row(
            session_id=str(user_id),
            t_ms=int(time.time() * 1000),
            label=-1,  # unknown in live runs; you can relabel later if needed
            kb_trust=float(kb_trust) if kb_trust is not None else 0.0,
            mouse_trust=float(mouse_trust) if mouse_trust is not None else 0.0,
            behavioural_trust=float(behaviour_trust) if behaviour_trust is not None else 0.0,
            face_trust=float(face_trust) if face_trust is not None else 0.0,
            fused_trust=float(fused_for_log) if fused_for_log is not None else 0.0,
            action=str(action) if action is not None else "UNKNOWN",
        )
    except Exception as e:
        # Do not break the live API if logging fails
        log.exception("append_trust_row failed: %s", e)
    # ---------------------------------------------------------

    return JsonResponse(
        {
            "ok": True,
            "user_id": user_id,
            "kb_trust": kb_trust,
            "mouse_trust": mouse_trust,
            "face_match": face_match,
            "liveness": liveness,
            "behaviour_trust": behaviour_trust,
            "face_trust": face_trust,
            "overall_trust": overall_trust,
            "overall_trust_rolling": overall_trust_rolling,
            "action": action,
        }
    )


# ---------------------------------------------------------------------
# Face endpoints
# ---------------------------------------------------------------------

@csrf_exempt
def face_enroll(request: HttpRequest) -> JsonResponse:
    """
    Enroll face template from a single base64 frame.
    """
    if request.method != "POST":
        return JsonResponse(
            {"ok": False, "error": "method_not_allowed", "detail": "POST required"},
            status=405,
        )

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception as e:
        return JsonResponse(
            {"ok": False, "error": "bad_json", "detail": str(e)}, status=400
        )

    img_b64 = (
        body.get("image")
        or body.get("image_b64")
        or body.get("frame")
        or body.get("img")
    )

    engine = _get_face_engine()
    res = engine.enroll_from_b64(img_b64)
    # res is already a dict with ok / error
    status = 200 if res.get("ok") else 400
    return JsonResponse(res, status=status)


@csrf_exempt
def face_score(request: HttpRequest) -> JsonResponse:
    """
    Score a single base64 frame for face_match + liveness.
    """
    if request.method != "POST":
        return JsonResponse(
            {"ok": False, "error": "method_not_allowed", "detail": "POST required"},
            status=405,
        )

    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception as e:
        return JsonResponse(
            {"ok": False, "error": "bad_json", "detail": str(e)}, status=400
        )

    img_b64 = (
        body.get("image")
        or body.get("image_b64")
        or body.get("frame")
        or body.get("img")
    )

    engine = _get_face_engine()
    res = engine.score_from_b64(img_b64)
    return JsonResponse(res)
