import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eval_utils import compute_roc_pr_eer, find_best_threshold_for_f1

LOG_CSV = "data/live_trust_timeseries.csv"
RESULTS_DIR = "results/fusion_live"

os.makedirs(RESULTS_DIR, exist_ok=True)


def eval_fusion_live():
    df = pd.read_csv(LOG_CSV)

    # Keep only labelled rows (0 or 1)
    df = df[df["label"].isin([0, 1])].copy()

    y = df["label"].astype(int).values

    # Scores we will compare
    score_sets = {
        "keyboard": df["kb_trust"].values,
        "mouse": df["mouse_trust"].values,
        "behavioural": df["behavioural_trust"].values,          # kb+mouse
        "behavioural_plus_face": df["fused_trust"].values,      # kb+mouse+face
    }

    summary_rows = []

    # ---------- ROC & PR curves ----------
    plt.figure()
    for name, scores in score_sets.items():
        roc = compute_roc_pr_eer(y, scores)
        thr_f1, metrics = find_best_threshold_for_f1(y, scores)

        summary_rows.append({
            "model": name,
            "threshold_best_f1": metrics.threshold,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "far": metrics.far,
            "frr": metrics.frr,
            "eer": roc.eer,
            "eer_threshold": roc.eer_threshold,
            "roc_auc": roc.roc_auc,
            "pr_auc": roc.pr_auc,
        })

        plt.plot(roc.fpr, roc.tpr, label=f"{name} (AUC={roc.roc_auc:.3f})")

    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC – keyboard vs mouse vs fusion vs fusion+face")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_fusion_roc.png"))
    plt.close()

    # ---------- PR curves ----------
    plt.figure()
    for name, scores in score_sets.items():
        roc = compute_roc_pr_eer(y, scores)
        plt.plot(
            roc.recall_curve,
            roc.precision_curve,
            label=f"{name} (AUC={roc.pr_auc:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR – keyboard vs mouse vs fusion vs fusion+face")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_fusion_pr.png"))
    plt.close()

    # ---------- Save summary table (for Table 4.5 / 4.6) ----------
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(RESULTS_DIR, "fusion_live_summary.csv"),
        index=False,
    )

    # ---------- Histogram for fused trust (Figure 4.6) ----------
    fused_scores = score_sets["behavioural_plus_face"]
    roc_fused = compute_roc_pr_eer(y, fused_scores)
    thr_f1, metrics_fused = find_best_threshold_for_f1(y, fused_scores)

    plt.figure()
    plt.hist(
        fused_scores[y == 1],
        bins=30,
        alpha=0.5,
        label="Genuine",
        density=True,
    )
    plt.hist(
        fused_scores[y == 0],
        bins=30,
        alpha=0.5,
        label="Impostor",
        density=True,
    )

    # Use your policy thresholds here if you have them hard-coded
    # Example: upper = 0.7, lower = 0.3
    ALLOW_THR = 0.7
    LOCK_THR = 0.3

    plt.axvline(LOCK_THR, color="red", linestyle="--", label="LOCK threshold")
    plt.axvline(ALLOW_THR, color="green", linestyle="--", label="ALLOW threshold")
    plt.axvline(thr_f1, color="black", linestyle=":", label="Best F1 threshold")

    plt.xlabel("Fused trust score")
    plt.ylabel("Density")
    plt.title("Fused behavioural + face trust – Genuine vs impostor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_fused_hist.png"))
    plt.close()

    # ---------- Example time-series for one session (Figure 4.8) ----------
    # pick the first labelled session
    example_session = df["session_id"].iloc[0]
    df_sess = df[df["session_id"] == example_session].copy()
    df_sess = df_sess.sort_values("t_ms")

    plt.figure(figsize=(10, 4))
    plt.plot(df_sess["t_ms"], df_sess["kb_trust"], label="Keyboard trust")
    plt.plot(df_sess["t_ms"], df_sess["mouse_trust"], label="Mouse trust")
    plt.plot(df_sess["t_ms"], df_sess["behavioural_trust"], label="Behavioural (kb+mouse)")
    plt.plot(df_sess["t_ms"], df_sess["fused_trust"], label="Fused (behavioural+face)")

    plt.axhline(ALLOW_THR, color="green", linestyle="--", label="ALLOW threshold")
    plt.axhline(LOCK_THR, color="red", linestyle="--", label="LOCK threshold")

    plt.xlabel("Time (ms)")
    plt.ylabel("Trust")
    plt.title(f"Trust time-series for session {example_session}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_timeseries_example_session.png"))
    plt.close()

    print("Fusion/face evaluation complete. See:", RESULTS_DIR)


if __name__ == "__main__":
    eval_fusion_live()
