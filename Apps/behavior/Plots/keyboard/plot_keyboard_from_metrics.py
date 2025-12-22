import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

HERE = os.path.abspath(os.path.dirname(__file__))          # .../Apps/behavior/Plots/keyboard
# Go FOUR levels up: keyboard -> Plots -> behavior -> Apps -> H1
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))  # .../H1

CAE_CSV = os.path.join(PROJECT_ROOT, "artifacts", "kb_cae_metrics.csv")
SVM_CSV = os.path.join(PROJECT_ROOT, "artifacts", "kb_svm_eval_metrics.csv")

OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "keyboard")
os.makedirs(OUT_DIR, exist_ok=True)


def add_derived_metrics(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    From CSV with columns:
        user_id, FAR, FRR, ACC, n_genuine, n_impostor
    compute TP, FP, TN, FN, precision, recall, F1.
    """
    df = df.copy()

    P = df["n_genuine"].astype(float)
    N = df["n_impostor"].astype(float)
    FAR = df["FAR"].astype(float)
    FRR = df["FRR"].astype(float)

    FP = FAR * N
    FN = FRR * P
    TP = P - FN
    TN = N - FP

    precision = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
    recall = np.where(P > 0, TP / P, 0.0)
    f1 = np.where(
        precision + recall > 0,
        2.0 * precision * recall / (precision + recall),
        0.0,
    )

    df["TP"] = TP
    df["FP"] = FP
    df["TN"] = TN
    df["FN"] = FN
    df["precision"] = precision
    df["recall"] = recall
    df["f1"] = f1
    df["model"] = model_name

    return df


def macro_summary(df: pd.DataFrame) -> dict:
    return {
        "FAR_macro": df["FAR"].mean(),
        "FRR_macro": df["FRR"].mean(),
        "ACC_macro": df["ACC"].mean(),
        "F1_macro": df["f1"].mean(),
    }


def main():
    if not os.path.exists(CAE_CSV):
        raise FileNotFoundError(f"Cannot find CAE metrics CSV: {CAE_CSV}")
    if not os.path.exists(SVM_CSV):
        raise FileNotFoundError(f"Cannot find SVM metrics CSV: {SVM_CSV}")

    cae_raw = pd.read_csv(CAE_CSV)
    svm_raw = pd.read_csv(SVM_CSV)

    cae = add_derived_metrics(cae_raw, "CAE")
    svm = add_derived_metrics(svm_raw, "SVM")

    # ---- Global (macro) summary for Table 4.2 ----
    cae_macro = macro_summary(cae)
    svm_macro = macro_summary(svm)

    summary_rows = [
        {"model": "keyboard_AE", **cae_macro},
        {"model": "keyboard_SVM", **svm_macro},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "keyboard_models_global_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("=== Keyboard models macro summary (Table 4.2) ===")
    print(summary_df.to_string(index=False))
    print(f"[OK] Wrote global summary -> {summary_path}")

    # ---- Per-user SVM bar chart (Figure 4.2) ----
    svm_sorted = svm.sort_values("f1", ascending=False)

    plt.figure(figsize=(10, 4))
    plt.bar(svm_sorted["user_id"], svm_sorted["f1"])
    plt.xticks(rotation=90)
    plt.ylim(0.0, 1.0)
    plt.xlabel("User ID")
    plt.ylabel("F1-score")
    plt.title("Per-user keyboard SVM F1-score")
    plt.tight_layout()

    f1_fig = os.path.join(OUT_DIR, "keyboard_svm_per_user_f1.png")
    plt.savefig(f1_fig)
    plt.close()
    print(f"[OK] Wrote per-user SVM F1 plot -> {f1_fig}")

    # ---- Optional: AE vs SVM FAR/FRR comparison ----
    labels = ["FAR", "FRR"]
    cae_vals = [cae_macro["FAR_macro"], cae_macro["FRR_macro"]]
    svm_vals = [svm_macro["FAR_macro"], svm_macro["FRR_macro"]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, cae_vals, width, label="AE")
    plt.bar(x + width / 2, svm_vals, width, label="SVM")
    plt.xticks(x, labels)
    plt.ylabel("Rate")
    plt.ylim(0.0, 1.0)
    plt.title("Keyboard AE vs SVM – macro FAR/FRR")
    plt.legend()
    plt.tight_layout()

    comp_fig = os.path.join(OUT_DIR, "keyboard_ae_vs_svm_far_frr.png")
    plt.savefig(comp_fig)
    plt.close()
    print(f"[OK] Wrote AE vs SVM FAR/FRR plot -> {comp_fig}")


if __name__ == "__main__":
    main()
