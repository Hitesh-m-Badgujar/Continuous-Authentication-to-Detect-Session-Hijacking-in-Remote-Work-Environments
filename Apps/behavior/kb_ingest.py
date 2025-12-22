# Apps/behavior/kb_ingest.py
import pandas as pd
import numpy as np
from pathlib import Path
import glob, os, csv

# ----- Windowing over strokes (not time) -----
WIN = 240   # strokes per window
HOP = 40   # shift in strokes

FEATURES = [
    "ks_count","ks_unique",
    "dwell_mean","dwell_std","dwell_p10","dwell_p50","dwell_p90",
    "dd_mean","dd_std","dd_p10","dd_p50","dd_p90",
    "ud_mean","ud_std","ud_p10","ud_p50","ud_p90",
    "backspace_rate","burst_mean","idle_frac"
]

def _pct(x, q):
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) else 0.0

def _stats(x):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        float(np.mean(x)),
        float(np.std(x)),
        _pct(x, 10),
        _pct(x, 50),
        _pct(x, 90),
    )

def _burst_from_dd(dd):
    """Average run-length of short gaps (<150 ms) using DD (sec)."""
    dd = dd[~np.isnan(dd)]
    if len(dd) == 0:
        return 0.0
    flags = dd < 0.150
    run = 0
    runs = []
    for b in flags:
        if b:
            run += 1
        elif run > 0:
            runs.append(run)
            run = 0
    if run > 0:
        runs.append(run)
    return float(np.mean(runs)) if runs else 0.0

def _idle_from_ud(ud):
    """Fraction of big gaps (>1.0 s) using UD (sec)."""
    ud = ud[~np.isnan(ud)]
    if len(ud) == 0:
        return 1.0
    return float(np.mean(ud > 1.0))

def read_csv_robust(path: str) -> pd.DataFrame | None:
    """Read CSV while skipping malformed rows and handling encodings."""
    try:
        return pd.read_csv(
            path,
            engine="python",
            on_bad_lines="skip",
            quoting=csv.QUOTE_MINIMAL,
        )
    except Exception:
        try:
            return pd.read_csv(
                path,
                engine="python",
                on_bad_lines="skip",
                quoting=csv.QUOTE_MINIMAL,
                encoding="latin1",
            )
        except Exception as e2:
            print(f"SKIP (cannot parse): {path} -> {e2}")
            return None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected column names to a common set."""
    # Lowercase map
    lower = {c.lower(): c for c in df.columns}
    # Possible aliases
    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    col_user   = pick("user_id","userid","user")
    col_sess   = pick("session_id","sessionid","session")
    col_key    = pick("key_pressed","key","keycode")
    col_hold   = pick("hold_time","hold","dwell","dwell_time")
    col_dd     = pick("dd","down_down","down-down")
    col_ud     = pick("ud","up_down","up-down")

    needed = [col_user, col_sess, col_key, col_hold, col_dd, col_ud]
    if any(x is None for x in needed):
        missing = []
        if col_user is None: missing.append("User_ID")
        if col_sess is None: missing.append("Session_ID")
        if col_key is None: missing.append("Key_Pressed")
        if col_hold is None: missing.append("Hold_Time")
        if col_dd is None: missing.append("DD")
        if col_ud is None: missing.append("UD")
        raise KeyError(f"Missing expected columns (or aliases): {missing}")

    df_std = pd.DataFrame({
        "User_ID":    df[col_user].astype(str),
        "Session_ID": df[col_sess].astype(str),
        "Key_Pressed": df[col_key].astype(str),
        "Hold_Time":  pd.to_numeric(df[col_hold], errors="coerce")/1000.0 if df[col_hold].max() > 5 else pd.to_numeric(df[col_hold], errors="coerce"),
        "DD":         pd.to_numeric(df[col_dd],   errors="coerce"),
        "UD":         pd.to_numeric(df[col_ud],   errors="coerce"),
    })
    # Note: If Hold_Time was in ms (values > 5), convert to seconds.
    return df_std.reset_index(drop=True)

def process_keyboard_folder(root="Datasets/Keyboard") -> pd.DataFrame:
    files = glob.glob(os.path.join(root, "**", "*_keystroke_raw.csv"), recursive=True)
    if not files:
        # Fallback: any csv under Keyboard
        files = glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    if not files:
        raise SystemExit(f"No keyboard CSVs found under {root}")

    out_rows = []
    for f in files:
        df = read_csv_robust(f)
        if df is None or df.empty:
            continue
        try:
            df = standardize_columns(df)
        except Exception as e:
            print(f"SKIP (missing cols): {f} -> {e}")
            continue

        # Order as typed (assume file order is stroke order)
        df = df.reset_index(drop=True)

        uid = df["User_ID"].astype(str).iloc[0]
        sid = df["Session_ID"].astype(str).iloc[0]

        n = len(df)
        i = 0
        while i + WIN <= n:
            w = df.iloc[i:i+WIN]
            holds = w["Hold_Time"].astype(float).to_numpy()
            dds   = w["DD"].astype(float).to_numpy()
            uds   = w["UD"].astype(float).to_numpy()
            keys  = w["Key_Pressed"].astype(str).to_numpy()

            # Sanity clip for negatives / huge outliers
            holds = np.clip(holds, 0.005, 3.0) # seconds
            dds   = np.clip(dds,   0.005, 3.0)
            uds   = np.clip(uds,   0.005, 3.0)

            dwell_s = _stats(holds)
            dd_s    = _stats(dds)
            ud_s    = _stats(uds)

            backspace_rate = float(np.mean([1 if k.lower() in ("backspace","key.backspace") else 0 for k in keys]))
            burst_mean     = _burst_from_dd(dds)
            idle_frac      = _idle_from_ud(uds)

            row = {
                "user_id": uid,
                "session_id": sid,
                "start_idx": i,
                "end_idx": i+WIN,
                "ks_count": int(WIN),
                "ks_unique": int(len(set(keys))),
                "dwell_mean": dwell_s[0], "dwell_std": dwell_s[1], "dwell_p10": dwell_s[2], "dwell_p50": dwell_s[3], "dwell_p90": dwell_s[4],
                "dd_mean": dd_s[0],       "dd_std": dd_s[1],       "dd_p10": dd_s[2],       "dd_p50": dd_s[3],       "dd_p90": dd_s[4],
                "ud_mean": ud_s[0],       "ud_std": ud_s[1],       "ud_p10": ud_s[2],       "ud_p50": ud_s[3],       "ud_p90": ud_s[4],
                "backspace_rate": backspace_rate,
                "burst_mean": burst_mean,
                "idle_frac": idle_frac
            }
            out_rows.append(row)
            i += HOP

    out = pd.DataFrame(out_rows)
    return out

if __name__ == "__main__":
    out = process_keyboard_folder("Datasets/Keyboard")
    Path("Data").mkdir(exist_ok=True, parents=True)
    out.to_csv("Data/kb_windows.csv", index=False)
    print("WROTE Data/kb_windows.csv with", len(out), "rows for", out["user_id"].nunique(), "users")
    print(out.head(5).to_string(index=False))
