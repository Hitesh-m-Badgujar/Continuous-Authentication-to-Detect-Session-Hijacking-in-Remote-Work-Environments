import pandas as pd
import numpy as np
import os

SRC = "Data/keyboard_test_windows.csv"
OUT = "Data/keyboard_test_labeled.csv"

def main():
    df = pd.read_csv(SRC)

    if "user_id" not in df.columns:
        raise ValueError("keyboard_test_windows.csv does NOT contain user_id column.")

    users = sorted(df["user_id"].unique())

    print("[INFO] Found users:", users)

    all_rows = []

    for u in users:
        df_u = df[df["user_id"] == u].copy()
        df_u["label"] = 1     # genuine windows

        # impostor windows = windows from other users
        df_imp = df[df["user_id"] != u].copy()
        df_imp = df_imp.sample(len(df_u), replace=True)   # balance sizes
        df_imp["label"] = 0

        all_rows.append(df_u)
        all_rows.append(df_imp)

    out_df = pd.concat(all_rows, ignore_index=True)
    out_df.to_csv(OUT, index=False)

    print("[OK] Wrote labeled keyboard test file →", OUT)
    print("[INFO] Rows:", len(out_df))

if __name__ == "__main__":
    main()
