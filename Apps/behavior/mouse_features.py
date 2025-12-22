#!/usr/bin/env python3
"""
Mouse session feature extraction.

Expected normalized columns in input DataFrame:
  ['t','x','y','button','state','wheel','pressure']

- t: timestamp (s or ms; we auto-normalize)
- x,y: positions (pixels)
- button/state/wheel/pressure: optional; absent -> filled with defaults

Output features (FEATURE_ORDER) are numeric and stable for ML.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------- Feature inventory ------------------------------

FEATURE_ORDER: List[str] = [
    # basic kinematics
    "duration_s",
    "n_events",
    "path_length_px",
    "net_displacement_px",
    "straightness",            # net / path (0..1)
    "speed_mean",
    "speed_std",
    "speed_max",
    "accel_mean",
    "accel_std",
    "jerk_mean",
    "jerk_std",
    # angle / curvature
    "curv_mean",
    "curv_std",
    # pauses
    "pause_count_ge_100ms",
    "pause_mean_ms",
    # clicks / wheel (defensive: zeros if absent)
    "left_clicks",
    "right_clicks",
    "click_rate_hz",
    "wheel_events",
]


# ------------------------------ Utilities -------------------------------------

def save_feature_schema(path: Path | str, feature_order: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump({"feature_order": feature_order}, f, indent=2)


def sniff_csv(path: Path) -> dict:
    """
    Conservative CSV sniffing. We let pandas guess the separator when possible.
    """
    name = path.name.lower()
    # many academic sets are comma or tab separated
    return dict(engine="python", sep=None, encoding="utf-8", dtype=str, on_bad_lines="skip")


def _to_float(a: pd.Series) -> np.ndarray:
    # robust float cast, coerce errors to nan, then fill forward/backward
    v = pd.to_numeric(a, errors="coerce").astype(float)
    return v.to_numpy()


def normalize_mouse_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Map messy column names onto a standard schema: t,x,y,button,state,wheel,pressure
    Returns None if we do not have at least t,x,y.
    """
    cols = {c.strip().lower(): c for c in df.columns}
    # try common synonyms
    t_keys = ["t", "time", "timestamp", "client timestamp", "event_time", "ts"]
    x_keys = ["x", "xpos", "x coordinate", "client x", "pos_x"]
    y_keys = ["y", "ypos", "y coordinate", "client y", "pos_y"]
    btn_keys = ["button", "buttons", "btn"]
    state_keys = ["state", "event", "action"]         # press/release/move
    wheel_keys = ["wheel", "scroll", "wheel_delta"]
    press_keys = ["pressure", "force"]

    def pick(keys):
        for k in keys:
            if k in cols:
                return cols[k]
        return None

    t_col = pick(t_keys)
    x_col = pick(x_keys)
    y_col = pick(y_keys)

    if t_col is None or x_col is None or y_col is None:
        # cannot use this file
        return None

    out = pd.DataFrame({
        "t": _to_float(df[t_col]),
        "x": _to_float(df[x_col]),
        "y": _to_float(df[y_col]),
    })

    b_col = pick(btn_keys)
    s_col = pick(state_keys)
    w_col = pick(wheel_keys)
    p_col = pick(press_keys)

    out["button"] = (_to_float(df[b_col]) if b_col else np.zeros(len(out)))
    out["state"] = (_to_float(df[s_col]) if s_col else np.zeros(len(out)))
    out["wheel"] = (_to_float(df[w_col]) if w_col else np.zeros(len(out)))
    out["pressure"] = (_to_float(df[p_col]) if p_col else np.zeros(len(out)))

    # drop rows with any of x/y/t missing
    out = out.dropna(subset=["t", "x", "y"])
    if len(out) < 5:
        return None

    # normalize time: if looks like ms, convert to seconds
    t = out["t"].to_numpy()
    if np.nanmax(np.diff(np.sort(t))[:100]) > 20:  # terrible heuristic for ms scale
        out["t"] = out["t"] / 1000.0

    # ensure monotonic time (drop backwards jumps)
    out = out.sort_values("t")
    out = out.loc[out["t"].diff().fillna(0) >= 0]

    return out.reset_index(drop=True)


# ---------------------------- Feature extraction ------------------------------

def _finite_diff(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    dt = np.diff(t)
    dt[dt == 0] = np.nan
    dv = np.diff(v)
    a = dv / dt
    # align size by padding with first element
    a = np.concatenate([[a[0] if len(a) else 0.0], a])
    return a


def _angle_series(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return np.arctan2(dy, dx)


def _curvature(angles: np.ndarray, t: np.ndarray) -> np.ndarray:
    # angular velocity magnitude as curvature proxy
    dth = np.diff(angles)
    dt = np.diff(t)
    dt[dt == 0] = np.nan
    k = np.abs(dth / dt)
    if len(k) == 0:
        return np.array([0.0])
    return np.concatenate([[k[0]], k])


def session_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute a single vector of features for one session/file.
    All outputs are finite floats (NaNs replaced by 0.0).
    """
    t = df["t"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    n = len(t)
    duration = float(np.nanmax(t) - np.nanmin(t)) if n > 1 else 0.0

    # path stats
    dx = np.diff(x)
    dy = np.diff(y)
    step = np.hypot(dx, dy)
    path_len = float(np.nansum(step))
    net_disp = float(np.hypot(x[-1] - x[0], y[-1] - y[0])) if n > 1 else 0.0
    straightness = float(net_disp / path_len) if path_len > 0 else 0.0

    # kinematics
    # speed = ds/dt
    dt = np.diff(t)
    dt[dt == 0] = np.nan
    speed = np.divide(step, dt, out=np.zeros_like(step), where=np.isfinite(dt))
    accel = _finite_diff(speed, t[:-1] if len(t) > 1 else np.array([0.0]))
    jerk = _finite_diff(accel, t)

    # curvature proxy
    angles = _angle_series(dx, dy)
    curv = _curvature(angles, t)

    # pauses (>= 100 ms gaps)
    gaps = np.diff(t)
    pause_mask = gaps >= 0.100
    pause_count = int(np.nansum(pause_mask))
    pause_mean = float(np.nanmean(gaps[pause_mask])) * 1000.0 if pause_count > 0 else 0.0

    # clicks / wheel (defensive)
    left = 0
    right = 0
    wheel_ev = 0
    if "button" in df.columns and "state" in df.columns:
        # crude: count transitions where state changes from 0->1 per button id
        b = df["button"].to_numpy()
        s = df["state"].to_numpy()
        # left=1,right=2 in many logs; treat any >0 as click if state rising
        rising = (np.diff(s) > 0).astype(int)
        if len(rising) > 0:
            left = int(np.sum((b[:-1] == 1) & (rising == 1)))
            right = int(np.sum((b[:-1] == 2) & (rising == 1)))
    if "wheel" in df.columns:
        w = df["wheel"].to_numpy()
        wheel_ev = int(np.sum(np.abs(w) > 0))

    click_rate = float((left + right) / duration) if duration > 0 else 0.0

    feats = {
        "duration_s": duration,
        "n_events": float(n),
        "path_length_px": path_len,
        "net_displacement_px": net_disp,
        "straightness": straightness,
        "speed_mean": float(np.nanmean(speed)) if speed.size else 0.0,
        "speed_std": float(np.nanstd(speed)) if speed.size else 0.0,
        "speed_max": float(np.nanmax(speed)) if speed.size else 0.0,
        "accel_mean": float(np.nanmean(accel)) if accel.size else 0.0,
        "accel_std": float(np.nanstd(accel)) if accel.size else 0.0,
        "jerk_mean": float(np.nanmean(jerk)) if jerk.size else 0.0,
        "jerk_std": float(np.nanstd(jerk)) if jerk.size else 0.0,
        "curv_mean": float(np.nanmean(curv)) if curv.size else 0.0,
        "curv_std": float(np.nanstd(curv)) if curv.size else 0.0,
        "pause_count_ge_100ms": float(pause_count),
        "pause_mean_ms": pause_mean,
        "left_clicks": float(left),
        "right_clicks": float(right),
        "click_rate_hz": click_rate,
        "wheel_events": float(wheel_ev),
    }

    # sanitize NaNs/Infs
    for k, v in list(feats.items()):
        if not np.isfinite(v):
            feats[k] = 0.0

    # ensure full vector
    for k in FEATURE_ORDER:
        if k not in feats:
            feats[k] = 0.0

    return feats
