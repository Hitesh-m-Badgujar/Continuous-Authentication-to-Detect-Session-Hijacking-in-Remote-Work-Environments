from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "Data"

SRC_CSV = DATA_DIR / "cmu_keystroke.csv"
OUT_CSV = DATA_DIR / "kb_cmu_windows.csv"

FEATURES: List[str] = [
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


def _percentiles(arr: np.ndarray, qs=(10, 50, 90)) -> list[float]:
    return [float(np.percentile(arr, q)) for q in qs]


def main() -> None:
    if not SRC_CSV.exists():
        raise SystemExit(f"Source CMU CSV not found: {SRC_CSV}")

    df = pd.read_csv(SRC_CSV)

    if "subject" not in df.columns:
        raise SystemExit("Expected 'subject' column in CMU CSV (check file format).")

    dwell_cols = [c for c in df.columns if c.startswith("H.")]
    dd_cols = [c for c in df.columns if c.startswith("DD.")]
    ud_cols = [c for c in df.columns if c.startswith("UD.")]

    if not dwell_cols or not dd_cols or not ud_cols:
        raise SystemExit(
            "Could not find H./DD./UD. columns in CMU CSV. "
            "Use DSL-StrongPasswordData / keystroke.csv format."
        )

    out_rows = []

    for _, row in df.iterrows():
        user_id = str(row["subject"]).strip()
        session = int(row.get("sessionIndex", 0))
        rep = int(row.get("rep", 0))

        dwell = row[dwell_cols].to_numpy(dtype=float)
        dd = row[dd_cols].to_numpy(dtype=float)
        ud = row[ud_cols].to_numpy(dtype=float)

        if not np.all(np.isfinite(dwell)) or not np.all(np.isfinite(dd)) or not np.all(
            np.isfinite(ud)
        ):
            continue

        d_mean = float(dwell.mean())
        d_std = float(dwell.std())
        d_p10, d_p50, d_p90 = _percentiles(dwell)

        dd_mean = float(dd.mean())
        dd_std = float(dd.std())
        dd_p10, dd_p50, dd_p90 = _percentiles(dd)

        ud_mean = float(ud.mean())
        ud_std = float(ud.std())
        ud_p10, ud_p50, ud_p90 = _percentiles(ud)

        backspace_rate = 0.0
        burst_mean = len(dwell)
        idle_frac = 0.0

        out_rows.append(
            {
                "user_id": user_id,
                "session_id": session,
                "window_id": rep,
                "dwell_mean": d_mean,
                "dwell_std": d_std,
                "dwell_p10": d_p10,
                "dwell_p50": d_p50,
                "dwell_p90": d_p90,
                "dd_mean": dd_mean,
                "dd_std": dd_std,
                "dd_p10": dd_p10,
                "dd_p50": dd_p50,
                "dd_p90": dd_p90,
                "ud_mean": ud_mean,
                "ud_std": ud_std,
                "ud_p10": ud_p10,
                "ud_p50": ud_p50,
                "ud_p90": ud_p90,
                "backspace_rate": backspace_rate,
                "burst_mean": float(burst_mean),
                "idle_frac": idle_frac,
            }
        )

    if not out_rows:
        raise SystemExit("No usable rows extracted from CMU CSV.")

    out_df = pd.DataFrame(out_rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print(f"Wrote CMU windows -> {OUT_CSV}")
    print(f"Rows: {len(out_df)}, users: {out_df['user_id'].nunique()}")


if __name__ == "__main__":
    main()
