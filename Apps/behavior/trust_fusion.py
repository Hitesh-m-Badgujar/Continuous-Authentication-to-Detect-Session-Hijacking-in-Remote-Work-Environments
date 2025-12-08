# Apps/behavior/trust_fusion.py

from __future__ import annotations

from typing import Optional
import numpy as np

LOCK = "LOCK"
STEP_UP = "STEP_UP"
ALLOW = "ALLOW"


def _clamp01(x: Optional[float]) -> Optional[float]:
    """Clamp to [0,1] or return None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------
# Behaviour fusion: keyboard + mouse
#   - Keyboard is primary (better model).
#   - Mouse can ONLY push trust up, never down.
# ---------------------------------------------------------------------
def fuse_behaviour(
    kb_trust: Optional[float],
    mouse_trust: Optional[float],
) -> Optional[float]:
    kb = _clamp01(kb_trust)
    ms = _clamp01(mouse_trust)

    if kb is None and ms is None:
        return None

    # both present → blend but don't let mouse reduce keyboard trust
    if kb is not None and ms is not None:
        # soft blend (20% weight to mouse)
        raw = 0.8 * kb + 0.2 * ms
        # guarantee: behaviour trust is NEVER lower than keyboard trust
        fused = max(kb, raw)
        return fused

    # only one modality present
    return kb if kb is not None else ms


# ---------------------------------------------------------------------
# Face fusion: face_match + liveness
# ---------------------------------------------------------------------
def fuse_face(
    face_match: Optional[float],
    liveness: Optional[float],
) -> Optional[float]:
    fm = _clamp01(face_match)
    lv = _clamp01(liveness)

    if fm is None and lv is None:
        return None

    if fm is not None and lv is not None:
        # Face match more important than liveness
        return 0.7 * fm + 0.3 * lv

    return fm if fm is not None else lv


# ---------------------------------------------------------------------
# Global fusion: behaviour + face
# ---------------------------------------------------------------------
def fuse_overall(
    behaviour_trust: Optional[float],
    face_trust: Optional[float],
) -> Optional[float]:
    bt = _clamp01(behaviour_trust)
    ft = _clamp01(face_trust)

    if bt is None and ft is None:
        return None

    if bt is not None and ft is not None:
        # 60% behaviour (kb+mouse), 40% face
        return 0.6 * bt + 0.4 * ft

    return bt if bt is not None else ft


# ---------------------------------------------------------------------
# Policy mapping
#   LOCK   : low trust
#   STEP_UP: medium trust
#   ALLOW  : high trust
# ---------------------------------------------------------------------
def trust_policy_action(trust: Optional[float]) -> str:
    t = _clamp01(trust)
    if t is None:
        return LOCK

    # thresholds slightly relaxed so you get more ALLOW when behaviour+face agree
    if t < 0.35:
        return LOCK
    if t < 0.60:
        return STEP_UP
    return ALLOW
