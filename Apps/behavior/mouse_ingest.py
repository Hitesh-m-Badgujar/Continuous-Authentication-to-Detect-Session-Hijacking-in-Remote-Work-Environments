# Apps/behavior/mouse_ingest.py
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np

# ---------------------------
# Feature schema
# ---------------------------
FEATURE_ORDER = [
    "dur_ms",
    "n_points",
    "path_len",
    "straight_len",
    "straightness",      # straight_len / path_len
    "mean_speed",
    "p95_speed",
    "max_speed",
    "mean_acc",
    "p95_acc",
    "max_acc",
    "mean_jerk",
    "p95_jerk",
    "max_jerk",
    "dx",
    "dy",
    "abs_dx",
    "abs_dy",
    "bbox_w",
    "bbox_h",
    "bbox_area",
    "direction_changes", # sign changes of dx, dy combined
    "pause_ratio_20ms",  # fraction of dt>20ms
]

def save_feature_schema(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"feature_order": FEATURE_ORDER}, f, indent=2)

# ---------------------------
# Robust session loader
# ---------------------------
def _read_balabit_csv(p: Path) -> Optional[pd.DataFrame]:
    """
    Read a Balabit session CSV with unknown delimiters/headers.
    Expect at least time(ms), x, y columns. Extra columns tolerated.
    Returns df with columns: t (float ms), x (float), y (float).
    """
    if not p.exists() or p.stat().st_size == 0:
        return None

    # Try common formats quickly
    candidates = [
        dict(sep=",", header=None),
        dict(sep=",", header=0),
        dict(sep=";", header=None),
        dict(sep=r"\s+", header=None),
    ]

    for kw in candidates:
        try:
            df = pd.read_csv(p, engine="python", **kw)
            # pick first 3 numeric columns
            num = df.select_dtypes(include=[np.number])
            if num.shape[1] < 3:
                # try to coerce all to numeric and re-check
                df2 = df.apply(pd.to_numeric, errors="coerce")
                num = df2.select_dtypes(include=[np.number])
            if num.shape[1] < 3:
                continue
            num = num.iloc[:, :3].copy()
            num.columns = ["t", "x", "y"]
            # drop NaNs
            num = num.dropna()
            if len(num) < 3:
                continue
            # Ensure sorted by time
            num = num.sort_values("t").reset_index(drop=True)
            # Remove duplicate time stamps
            num = num.loc[~num["t"].duplicated()].reset_index(drop=True)
            return num
        except Exception:
            continue
    return None

# ---------------------------
# Low-level kinematics
# ---------------------------
def _kinematics(df: pd.DataFrame) -> Dict[str, float]:
    t = df["t"].to_numpy(dtype=float)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    n = len(t)

    # dt in seconds
    dt = np.diff(t) / 1000.0
    # guard against zeros/negatives
    dt[dt <= 1e-6] = 1e-6

    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.hypot(dx, dy)

    v = dist / dt
    a = np.diff(v) / dt[1:] if len(v) > 1 else np.array([])
    j = np.diff(a) / dt[2:] if len(a) > 1 else np.array([])

    path_len = float(dist.sum())
    dx_total = float(x[-1] - x[0])
    dy_total = float(y[-1] - y[0])
    straight_len = float(math.hypot(dx_total, dy_total))
    straightness = float(straight_len / path_len) if path_len > 1e-9 else 0.0

    mean_speed = float(v.mean()) if v.size else 0.0
    p95_speed = float(np.percentile(v, 95)) if v.size else 0.0
    max_speed = float(v.max()) if v.size else 0.0

    mean_acc = float(a.mean()) if a.size else 0.0
    p95_acc = float(np.percentile(a, 95)) if a.size else 0.0
    max_acc = float(a.max()) if a.size else 0.0

    mean_jerk = float(j.mean()) if j.size else 0.0
    p95_jerk = float(np.percentile(j, 95)) if j.size else 0.0
    max_jerk = float(j.max()) if j.size else 0.0

    bbox_w = float(x.max() - x.min())
    bbox_h = float(y.max() - y.min())
    bbox_area = float(bbox_w * bbox_h)

    # direction change: sign changes for dx and dy
    sdx = np.sign(dx); sdy = np.sign(dy)
    dc = int((np.diff(sdx) != 0).sum() + (np.diff(sdy) != 0).sum()) if len(dx) > 1 else 0

    pause_ratio_20ms = float((dt > 0.02).mean()) if dt.size else 0.0

    return {
        "dur_ms": float(t[-1] - t[0]),
        "n_points": float(n),
        "path_len": path_len,
        "straight_len": straight_len,
        "straightness": straightness,
        "mean_speed": mean_speed,
        "p95_speed": p95_speed,
        "max_speed": max_speed,
        "mean_acc": mean_acc,
        "p95_acc": p95_acc,
        "max_acc": max_acc,
        "mean_jerk": mean_jerk,
        "p95_jerk": p95_jerk,
        "max_jerk": max_jerk,
        "dx": dx_total,
        "dy": dy_total,
        "abs_dx": float(abs(dx_total)),
        "abs_dy": float(abs(dy_total)),
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_area": bbox_area,
        "direction_changes": float(dc),
        "pause_ratio_20ms": pause_ratio_20ms,
    }

def session_features(df: pd.DataFrame) -> Dict[str, float]:
    return _kinematics(df)

def window_features(df: pd.DataFrame, start_idx: int, end_idx: int) -> Dict[str, float]:
    slice_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    return _kinematics(slice_df)

# ---------------------------
# Sliding window helper
# ---------------------------
def make_windows(n: int, win: int, stride: int) -> List[Tuple[int, int]]:
    if n <= 0 or win < 3:
        return []
    if stride <= 0:
        stride = win
    inds = []
    i = 0
    while i + win <= n:
        inds.append((i, i + win))
        i += stride
    # include tail window if nothing captured and n>=3
    if not inds and n >= 3:
        inds.append((0, n))
    return inds

# ---------------------------
# Discovery
# ---------------------------
def find_sessions(root: Path) -> List[Tuple[str, Path]]:
    """
    Return list of (user_id, file_path) for files under <root>/user*/session_* (any ext).
    """
    items = []
    if not root.exists():
        return items
    for user_dir in sorted(root.glob("user*")):
        if not user_dir.is_dir():
            continue
        uid = user_dir.name.replace("user", "")
        for p in sorted(user_dir.glob("session_*")) + sorted(user_dir.glob("session_*.csv")):
            if p.is_file():
                items.append((f"user{uid}", p))
    return items

# ---------------------------
# Main ingest
# ---------------------------
@dataclass
class IngestSummary:
    files_found: int
    rows_written: int
    users: int
    bad_files: int
    out_csv: str
    bad_log: Optional[str]

def run_ingest(
    dataset_root: Path,
    out_csv: Path,
    schema_json: Optional[Path],
    bad_log: Optional[Path],
    mode: str,
    min_points: int,
    window_events: int,
    stride_events: int,
    loose: bool,
    verbose: bool,
) -> IngestSummary:
    pairs = find_sessions(dataset_root)
    if verbose:
        print(f"Found {len(pairs)} files under {dataset_root}")
        for uid, p in pairs[:10]:
            print(f"  - {p}")

    if not pairs:
        raise RuntimeError(f"No data files found under: {dataset_root}")

    rows = []
    bad = []

    for uid, path in pairs:
        df = _read_balabit_csv(path)
        if df is None:
            bad.append({"file": str(path), "reason": "unreadable"})
            if verbose:
                print(f"[SKIP] unreadable: {path}")
            continue

        if len(df) < min_points:
            bad.append({"file": str(path), "reason": f"too_few_points({len(df)})"})
            if verbose:
                print(f"[SKIP] too few points ({len(df)}): {path}")
            continue

        try:
            if mode == "session":
                feats = session_features(df)
                rows.append({"user_id": uid, "file": str(path), **feats})
            else:
                win = window_events
                stride = stride_events if stride_events > 0 else window_events
                for (a, b) in make_windows(len(df), win, stride):
                    if b - a < min_points:
                        continue
                    feats = window_features(df, a, b)
                    rows.append({
                        "user_id": uid,
                        "file": str(path),
                        "start_idx": a,
                        "end_idx": b,
                        **feats
                    })
        except Exception as e:
            if loose:
                bad.append({"file": str(path), "reason": f"feature_error:{type(e).__name__}"})
                if verbose:
                    print(f"[SKIP] feature error {path}: {e}")
            else:
                raise

    if not rows:
        raise RuntimeError("No valid rows produced. Try --loose and/or smaller --min-points/--window-events.")

    df_out = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    if bad_log:
        bad_log.parent.mkdir(parents=True, exist_ok=True)
        with open(bad_log, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "reason"])
            w.writeheader()
            w.writerows(bad)

    if schema_json:
        save_feature_schema(schema_json)

    users = df_out["user_id"].nunique() if "user_id" in df_out.columns else 0
    if verbose:
        print(f"WROTE {out_csv} with {len(df_out)} rows for {users} users (bad_files={len(bad)})")
        if bad_log:
            print(f"Bad files log -> {bad_log}")

    return IngestSummary(
        files_found=len(pairs),
        rows_written=len(df_out),
        users=users,
        bad_files=len(bad),
        out_csv=str(out_csv),
        bad_log=str(bad_log) if bad_log else None
    )

# ---------------------------
# CLI
# ---------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Ingest Balabit mouse sessions to feature CSV (session or window mode).")
    ap.add_argument("root", type=str, help="Path to 'training_files' or 'test_files' folder (containing user*/session_*).")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path.")
    ap.add_argument("--schema", type=str, default=None, help="Where to save feature schema JSON.")
    ap.add_argument("--bad-log", type=str, default=None, help="CSV to log bad/unreadable files.")
    ap.add_argument("--mode", choices=["session", "window"], default="window", help="Aggregate whole session or sliding windows.")
    ap.add_argument("--min-points", type=int, default=20, help="Minimum points required to keep a sample (after slicing).")
    ap.add_argument("--window-events", type=int, default=200, help="Events per window (window mode).")
    ap.add_argument("--stride-events", type=int, default=100, help="Stride between windows (window mode).")
    ap.add_argument("--loose", action="store_true", help="Skip problematic files instead of raising.")
    ap.add_argument("--verbose", action="store_true", help="Print details.")
    return ap.parse_args(argv)

def main():
    args = parse_args()

    root = Path(args.root)
    out_csv = Path(args.out)
    schema_json = Path(args.schema) if args.schema else None
    bad_log = Path(args.bad_log) if args.bad_log else None

    try:
        summary = run_ingest(
            dataset_root=root,
            out_csv=out_csv,
            schema_json=schema_json,
            bad_log=bad_log,
            mode=args.mode,
            min_points=args.min_points,
            window_events=args.window_events,
            stride_events=args.stride_events,
            loose=args.loose,
            verbose=args.verbose,
        )
        print(json.dumps({
            "ok": True,
            "files_found": summary.files_found,
            "rows_written": summary.rows_written,
            "users": summary.users,
            "bad_files": summary.bad_files,
            "out_csv": summary.out_csv,
            "bad_log": summary.bad_log,
        }, indent=2))
    except Exception as e:
        print("INGEST FAILED", file=sys.stderr)
        print(str(e), file=sys.stderr)
        print(json.dumps({"ok": False, "error": type(e).__name__, "detail": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
