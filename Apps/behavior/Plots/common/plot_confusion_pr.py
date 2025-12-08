# Apps/behavior/Plots/common/plot_confusion_pr.py

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)


def load_data(csv_path: str, label_col: str, score_col: str, pos_label: int):
    df = pd.read_csv(csv_path)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}. "
                         f"Available columns: {list(df.columns)}")
    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not found in {csv_path}. "
                         f"Available columns: {list(df.columns)}")

    y_true = df[label_col].values.astype(int)
    scores = df[score_col].values.astype(float)

    # Map labels to {0,1} with given positive label
    y_bin = (y_true == pos_label).astype(int)
    return y_bin, scores


def compute_confusion(y_true: np.ndarray,
                      scores: np.ndarray,
                      threshold: float):
    """
    Convert scores -> hard predictions using given threshold
    and compute confusion matrix + basic metrics.
    """
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # cm layout:
    # [[TP, FN],
    #  [FP, TN]]

    tn = cm[1, 1]
    fp = cm[1, 0]
    fn = cm[0, 1]
    tp = cm[0, 0]

    report = classification_report(
        y_true,
        y_pred,
        labels=[1, 0],
        target_names=["genuine(1)", "impostor(0)"],
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": (tp + tn) / max(1, (tp + tn + fp + fn)),
        "precision": report["genuine(1)"]["precision"],
        "recall": report["genuine(1)"]["recall"],
        "f1": report["genuine(1)"]["f1-score"],
        "far": fp / max(1, (fp + tn)),  # False Acceptance Rate
        "frr": fn / max(1, (fn + tp)),  # False Rejection Rate
    }

    return cm, metrics


def plot_confusion_matrix(cm: np.ndarray,
                          out_path: str,
                          title: str = "Confusion matrix"):
    """
    cm is 2x2, layout:
      [[TP, FN],
       [FP, TN]]
    """
    fig, ax = plt.subplots(figsize=(4, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Pred genuine", "Pred impostor"],
        yticklabels=["True genuine", "True impostor"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    # Write numbers
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray,
                  scores: np.ndarray,
                  out_path: str,
                  title: str = "Precision–Recall curve"):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"PR (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.grid(True)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot confusion matrix and PR curve from score CSV."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with labels and scores.",
    )
    parser.add_argument(
        "--label-col",
        required=True,
        help="Name of the label column (e.g., 'label').",
    )
    parser.add_argument(
        "--score-col",
        required=True,
        help="Name of the score column (continuous, higher = genuine).",
    )
    parser.add_argument(
        "--pos-label",
        type=int,
        default=1,
        help="Which value in label-col is the 'genuine' class (default: 1).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Decision threshold on the score to build confusion matrix.",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Prefix for output files (PNG + metrics JSON).",
    )

    args = parser.parse_args()

    y_true, scores = load_data(
        csv_path=args.csv,
        label_col=args.label_col,
        score_col=args.score_col,
        pos_label=args.pos_label,
    )

    cm, metrics = compute_confusion(
        y_true=y_true,
        scores=scores,
        threshold=args.threshold,
    )

    # Save confusion matrix plot
    cm_png = args.out_prefix + "_confusion.png"
    plot_confusion_matrix(
        cm,
        out_path=cm_png,
        title=os.path.basename(args.out_prefix) + " – confusion matrix",
    )

    # Save PR curve plot
    pr_png = args.out_prefix + "_pr.png"
    plot_pr_curve(
        y_true=y_true,
        scores=scores,
        out_path=pr_png,
        title=os.path.basename(args.out_prefix) + " – PR curve",
    )

    # Save metrics as a small CSV for reference
    metrics_csv = args.out_prefix + "_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    print("Saved:")
    print("  Confusion matrix:", cm_png)
    print("  PR curve:", pr_png)
    print("  Metrics CSV:", metrics_csv)


if __name__ == "__main__":
    main()
