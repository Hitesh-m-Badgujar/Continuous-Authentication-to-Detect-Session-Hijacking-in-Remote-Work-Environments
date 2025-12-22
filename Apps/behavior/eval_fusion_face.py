# Apps/behavior/eval_fusion_face.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
LIVE_TRUST_CSV = "Data/live_trust_sessions.csv"

os.makedirs(RESULTS_DIR, exist_ok=True)


def eval_face_fusion():
    df = pd.read_csv(LIVE_TRUST_CSV)

    # expected columns: is_genuine, trust_keyboard, trust_mouse,
    #                   trust_face, trust_fused
    df_genuine = df[df["is_genuine"] == 1]
    df_impostor = df[df["is_genuine"] == 0]

    # Figure 4.6 – histogram of fused trust for genuine vs impostor
    plt.figure()
    bins = np.linspace(0.0, 1.0, 30)
    plt.hist(
        df_genuine["trust_fused"],
        bins=bins,
        alpha=0.5,
        label="Genuine",
        density=True,
    )
    plt.hist(
        df_impostor["trust_fused"],
        bins=bins,
        alpha=0.5,
        label="Impostor",
        density=True,
    )
    # if you want, manually mark your ALLOW / STEP-UP / LOCK thresholds:
    # plt.axvline(0.75, color="black", linestyle="--", label="ALLOW threshold")
    # plt.axvline(0.40, color="red", linestyle="--", label="LOCK threshold")

    plt.xlabel("Fused trust score")
    plt.ylabel("Density")
    plt.title("Distribution of fused trust – genuine vs impostor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "fig_trust_histogram_fused.png"))
    plt.close()

    print("Face+behavioural fusion histogram saved.")


if __name__ == "__main__":
    eval_face_fusion()
