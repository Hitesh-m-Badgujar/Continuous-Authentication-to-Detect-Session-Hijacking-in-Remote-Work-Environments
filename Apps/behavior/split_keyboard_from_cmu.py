# Apps/behavior/split_keyboard_from_cmu.py

import os
import pandas as pd


def main():
    # This file is in H1/Apps/behavior -> go up 3 levels to get project root H1
    here = os.path.abspath(__file__)
    root = os.path.dirname(os.path.dirname(os.path.dirname(here)))  # .../H1
    data_dir = os.path.join(root, "Data")
    source_csv = os.path.join(data_dir, "kb_cmu_windows.csv")

    print(f"[INFO] Project root: {root}")
    print(f"[INFO] Loading keyboard windows from: {source_csv}")

    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Cannot find {source_csv}")

    df = pd.read_csv(source_csv)
    print("[INFO] Columns:", list(df.columns))

    # We expect both 'split' and 'label' to already be there
    if "split" not in df.columns:
        raise KeyError(
            "kb_cmu_windows.csv does not have a 'split' column.\n"
            "Open the file in Excel/VS Code and check which column "
            "indicates train/val/test; update this script if needed."
        )

    if "label" not in df.columns:
        raise KeyError(
            "kb_cmu_windows.csv does not have a 'label' column.\n"
            "Your AE/SVM eval scripts need genuine (1) vs impostor (0) labels."
        )

    # Just filter by split and WRITE ALL COLUMNS (including label)
    splits = [
        ("train", "keyboard_train_windows.csv"),
        ("val",   "keyboard_val_windows.csv"),
        ("test",  "keyboard_test_windows.csv"),
    ]

    for split_name, out_file in splits:
        sub = df[df["split"] == split_name].copy()
        out_path = os.path.join(data_dir, out_file)
        sub.to_csv(out_path, index=False)
        print(f"[OK] Wrote {split_name} CSV -> {out_path} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
