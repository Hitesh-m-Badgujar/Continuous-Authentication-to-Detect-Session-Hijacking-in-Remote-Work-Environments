from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


LOG_CSV_CANDIDATES = [
    "artifacts/realtime/live_trust_timeseries.csv",
    "Data/live_trust_timeseries.csv",
]
OUT_DIR = "artifacts/realtime"

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Column normalisation helpers
# ---------------------------------------------------------------------

def _ensure_required_csv() -> pd.DataFrame:
    log_csv = next((p for p in LOG_CSV_CANDIDATES if os.path.exists(p)), None)
    if log_csv is None:
        raise FileNotFoundError(
            "Cannot find any live trust log CSV. Checked: "
            + ", ".join(LOG_CSV_CANDIDATES)
        )

    df = pd.read_csv(log_csv)
    if df.empty:
        raise ValueError(f"{log_csv} exists but contains no rows")

    # session id --------------------------------------------------------
    if "session_id" not in df.columns:
        df["session_id"] = "session_1"

    # time column -------------------------------------------------------
    if "t_ms" not in df.columns:
        if "timestamp" in df.columns:
            # If timestamps are numeric-like, use them directly.
            try:
                df["t_ms"] = pd.to_numeric(df["timestamp"])
            except Exception:
                # Fallback: create synthetic evenly spaced ticks
                df["t_ms"] = range(len(df))
        else:
            df["t_ms"] = range(len(df))

    # behavioural trust aliases ----------------------------------------
    if "behavioural_trust" not in df.columns:
        if "behavioral_trust" in df.columns:
            df["behavioural_trust"] = df["behavioral_trust"]
        elif "behaviour_trust" in df.columns:
            df["behavioural_trust"] = df["behaviour_trust"]
        elif "behavior_trust" in df.columns:
            df["behavioural_trust"] = df["behavior_trust"]

    # face trust aliases / reconstruction -------------------------------
    if "face_trust" not in df.columns:
        if "face_match" in df.columns and "face_liveness" in df.columns:
            df["face_trust"] = 0.7 * df["face_match"] + 0.3 * df["face_liveness"]

    # fused trust aliases ----------------------------------------------
    if "fused_trust" not in df.columns:
        if "overall_trust" in df.columns:
            df["fused_trust"] = df["overall_trust"]
        elif "overall_trust_rolling" in df.columns:
            df["fused_trust"] = df["overall_trust_rolling"]

    # rolling fused aliases --------------------------------------------
    if "fused_trust_rolling" not in df.columns:
        if "overall_trust_rolling" in df.columns:
            df["fused_trust_rolling"] = df["overall_trust_rolling"]

    # mouse trust aliases ----------------------------------------------
    if "mouse_trust" not in df.columns and "mouse_score" in df.columns:
        df["mouse_trust"] = df["mouse_score"]

    # keyboard trust aliases -------------------------------------------
    if "kb_trust" not in df.columns and "keyboard_trust" in df.columns:
        df["kb_trust"] = df["keyboard_trust"]

    # action fallback ---------------------------------------------------
    if "action" not in df.columns:
        df["action"] = "UNKNOWN"

    needed_now = ["session_id", "t_ms", "kb_trust", "mouse_trust", "behavioural_trust", "fused_trust", "action"]
    missing = [c for c in needed_now if c not in df.columns]
    if missing:
        raise ValueError(
            f"{log_csv} is missing required columns even after alias handling: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    print(f"[OK] Using live trust log -> {log_csv}")
    return df


def _extract_latest_contiguous_session(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Pick the latest contiguous live session chunk.

    Why this is needed:
    - The runtime logger may keep appending rows across multiple monitor runs.
    - `session_id` is sometimes constant (e.g. 'global'), so grouping only by
      session_id mixes old and new recordings.

    We therefore split the log whenever:
    - time goes backwards, or
    - the time gap between adjacent rows is very large.
    """
    df = df.copy()
    df["t_ms_num"] = pd.to_numeric(df["t_ms"], errors="coerce").fillna(0)
    df = df.sort_values(["session_id", "t_ms_num"], kind="stable").reset_index(drop=True)

    # A new chunk starts if the session_id changes, if time goes backwards,
    # or if there is a large gap. 10 seconds is safely larger than the normal
    # 150 ms-ish update cadence.
    prev_session = df["session_id"].shift(1)
    prev_t = df["t_ms_num"].shift(1)
    gap = df["t_ms_num"] - prev_t

    new_chunk = (
        (df["session_id"] != prev_session)
        | (gap < 0)
        | (gap > 10_000)
        | prev_t.isna()
    )

    df["chunk_id"] = new_chunk.cumsum()
    last_chunk = int(df["chunk_id"].iloc[-1])
    df_chunk = df[df["chunk_id"] == last_chunk].copy()
    session_label = str(df_chunk["session_id"].iloc[0])

    # Drop helper cols before returning
    df_chunk = df_chunk.drop(columns=["t_ms_num", "chunk_id"], errors="ignore")
    return df_chunk, session_label


# ---------------------------------------------------------------------
# Main plotting routine
# ---------------------------------------------------------------------

def plot_timeseries_and_hist() -> None:
    df = _ensure_required_csv()

    # --- isolate the latest contiguous live session chunk ---
    df_sess, first_session = _extract_latest_contiguous_session(df)

    if df_sess.empty:
        raise ValueError("No rows found in the latest contiguous live session chunk")

    print(
        f"[OK] Using latest live chunk from session_id={first_session!r} "
        f"with {len(df_sess)} rows"
    )

    # time in seconds starting from 0
    t0 = pd.to_numeric(df_sess["t_ms"], errors="coerce").fillna(0).min()
    df_sess["t_sec"] = (pd.to_numeric(df_sess["t_ms"], errors="coerce").fillna(0) - t0) / 1000.0

    # ------------------------------------------------------------------
    # 1) Time-series plot of trust signals over time
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 4))

    def maybe_plot(col: str, label: str) -> None:
        if col in df_sess.columns:
            plt.plot(df_sess["t_sec"], df_sess[col], label=label)

    maybe_plot("kb_trust", "Keyboard trust")
    maybe_plot("mouse_trust", "Mouse trust")
    maybe_plot("behavioural_trust", "Behavioural (KB+Mouse)")
    maybe_plot("face_trust", "Face trust")
    maybe_plot("fused_trust", "Fused trust")
    maybe_plot("fused_trust_rolling", "Fused trust (rolling)")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Trust score")
    plt.title(f"Live trust time-series (session {first_session})")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path_ts = os.path.join(OUT_DIR, "fig_trust_timeseries.png")
    plt.savefig(out_path_ts, dpi=200)
    plt.close()
    print(f"[OK] Wrote time-series plot -> {out_path_ts}")

    # ------------------------------------------------------------------
    # 2) Histogram of trust distributions
    # ------------------------------------------------------------------
    plt.figure(figsize=(9, 5))

    hist_specs = [
        ("kb_trust", "Keyboard trust"),
        ("mouse_trust", "Mouse trust"),
        ("behavioural_trust", "Behavioural (KB+Mouse)"),
        ("face_trust", "Face trust"),
        ("fused_trust", "Fused trust"),
    ]

    plotted_any = False
    for col, label in hist_specs:
        if col in df_sess.columns:
            plt.hist(df_sess[col].dropna(), bins=30, alpha=0.45, label=label, density=True)
            plotted_any = True

    if not plotted_any:
        raise ValueError("No trust columns available for histogram plotting")

    plt.xlabel("Trust score")
    plt.ylabel("Density")
    plt.title(f"Trust score distributions (session {first_session})")
    plt.legend()
    plt.tight_layout()

    out_path_hist = os.path.join(OUT_DIR, "fig_trust_histograms.png")
    plt.savefig(out_path_hist, dpi=200)
    plt.close()
    print(f"[OK] Wrote histogram plot -> {out_path_hist}")

    # ------------------------------------------------------------------
    # 3) Bar chart of ALLOW / STEP_UP / LOCK counts
    # ------------------------------------------------------------------
    action_order = ["ALLOW", "STEP_UP", "LOCK"]
    action_counts = df_sess["action"].astype(str).value_counts()
    ordered_counts = pd.Series({a: int(action_counts.get(a, 0)) for a in action_order})

    # keep unknown actions too, but after the main three
    for action_name, count in action_counts.items():
        if action_name not in ordered_counts.index:
            ordered_counts.loc[action_name] = int(count)

    plt.figure(figsize=(5, 4))
    ordered_counts.plot(kind="bar")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title(f"Policy decisions over session {first_session}")
    plt.tight_layout()

    out_path_actions = os.path.join(OUT_DIR, "fig_actions_bar.png")
    plt.savefig(out_path_actions, dpi=200)
    plt.close()
    print(f"[OK] Wrote action bar plot -> {out_path_actions}")

    # ------------------------------------------------------------------
    # 4) Simple summary CSV for the report (means, std, min, max)
    # ------------------------------------------------------------------
    summary: Dict[str, Dict[str, float]] = {}
    for col in [
        "kb_trust",
        "mouse_trust",
        "behavioural_trust",
        "face_trust",
        "fused_trust",
        "fused_trust_rolling",
    ]:
        if col in df_sess.columns:
            s = df_sess[col].dropna().describe()
            summary[col] = {
                "mean": float(s["mean"]),
                "std": float(s["std"]),
                "min": float(s["min"]),
                "max": float(s["max"]),
            }

    summary_rows = []
    label_map = {
        "kb_trust": "keyboard",
        "mouse_trust": "mouse",
        "behavioural_trust": "behavioural",
        "face_trust": "face",
        "fused_trust": "fused",
        "fused_trust_rolling": "fused_rolling",
    }
    for col, stats in summary.items():
        row = {"signal": label_map.get(col, col)}
        row.update(stats)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    out_summary = os.path.join(OUT_DIR, "trust_summary.csv")
    summary_df.to_csv(out_summary, index=False)
    print(f"[OK] Wrote trust summary CSV -> {out_summary}")


if __name__ == "__main__":
    plot_timeseries_and_hist()
