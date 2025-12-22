# Apps/behavior/eval_realtime_plots.py

import os
import pandas as pd
import matplotlib.pyplot as plt


LOG_CSV = "Data/live_trust_timeseries.csv"
OUT_DIR = "artifacts/realtime"

os.makedirs(OUT_DIR, exist_ok=True)


def plot_timeseries_and_hist():
    if not os.path.exists(LOG_CSV):
        raise FileNotFoundError(f"Cannot find {LOG_CSV}")

    df = pd.read_csv(LOG_CSV)

    required_cols = [
        "session_id",
        "t_ms",
        "kb_trust",
        "mouse_trust",
        "behavioural_trust",
        "face_trust",
        "fused_trust",
        "action",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' in {LOG_CSV}, but it's missing.")

    # --- pick one session (usually you just have one: your own recording) ---
    # if there are multiple, this picks the first one
    first_session = str(df["session_id"].iloc[0])
    df_sess = df[df["session_id"] == first_session].copy()

    if df_sess.empty:
        raise ValueError(f"No rows found for session_id={first_session!r}")

    # time in seconds starting from 0
    t0 = df_sess["t_ms"].min()
    df_sess["t_sec"] = (df_sess["t_ms"] - t0) / 1000.0

    # ------------------------------------------------------------------
    # 1) Time-series plot of trust signals over time
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 4))

    def maybe_plot(col, label):
        if col in df_sess.columns:
            plt.plot(df_sess["t_sec"], df_sess[col], label=label)

    maybe_plot("kb_trust", "Keyboard trust")
    maybe_plot("mouse_trust", "Mouse trust")
    maybe_plot("behavioural_trust", "Behavioural (KB+Mouse)")
    maybe_plot("face_trust", "Face trust")
    maybe_plot("fused_trust", "Fused trust (Behavioural+Face)")

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
    # 2) Histogram of trust distributions (Behavioural vs Face vs Fused)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    if "behavioural_trust" in df_sess.columns:
        plt.hist(
            df_sess["behavioural_trust"],
            bins=30,
            alpha=0.6,
            label="Behavioural (KB+Mouse)",
            density=True,
        )

    if "face_trust" in df_sess.columns:
        plt.hist(
            df_sess["face_trust"],
            bins=30,
            alpha=0.6,
            label="Face",
            density=True,
        )

    if "fused_trust" in df_sess.columns:
        plt.hist(
            df_sess["fused_trust"],
            bins=30,
            alpha=0.6,
            label="Fused (Behavioural+Face)",
            density=True,
        )

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
    # 3) Bar chart of ALLOW / STEP_UP / LOCK counts for fused action
    # ------------------------------------------------------------------
    action_counts = df_sess["action"].value_counts().sort_index()

    plt.figure(figsize=(5, 4))
    action_counts.plot(kind="bar")
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
    summary = {}
    for col in ["kb_trust", "mouse_trust", "behavioural_trust", "face_trust", "fused_trust"]:
        if col in df_sess.columns:
            s = df_sess[col].describe()
            summary[col] = {
                "mean": float(s["mean"]),
                "std": float(s["std"]),
                "min": float(s["min"]),
                "max": float(s["max"]),
            }

    summary_rows = []
    for col, stats in summary.items():
        row = {"signal": col}
        row.update(stats)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    out_summary = os.path.join(OUT_DIR, "trust_summary.csv")
    summary_df.to_csv(out_summary, index=False)
    print(f"[OK] Wrote trust summary CSV -> {out_summary}")


if __name__ == "__main__":
    plot_timeseries_and_hist()
