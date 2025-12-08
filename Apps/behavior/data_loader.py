# H1/Apps/behavior/data_loader.py
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from django.conf import settings

def _dataset_csv() -> Path:
    base = Path(getattr(settings, "DATA_CSV", Path(settings.BASE_DIR) / "Data" / "aggregated.csv"))
    if not base.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {base}")
    return base

def load_aggregated(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(_dataset_csv())
    if "user_id" not in df.columns:
        raise ValueError("CSV must have a 'user_id' column.")
    if limit:
        keep_users = df["user_id"].drop_duplicates().iloc[:limit]
        df = df[df["user_id"].isin(keep_users)].copy()
    df.reset_index(drop=True, inplace=True)
    return df

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"user_id", "label", "target", "y"}
    cols = [c for c in df.columns if c not in drop and np.issubdtype(df[c].dtype, np.number)]
    if not cols:
        raise ValueError("No numeric feature columns found in dataset.")
    return cols

def per_user_splits(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42) -> Dict[str, Dict[str, pd.DataFrame]]:
    feat_cols = get_feature_columns(df)
    parts: Dict[str, Dict[str, pd.DataFrame]] = {}
    for uid, g in df.groupby("user_id"):
        g = g.copy()
        idx = np.arange(len(g))
        tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)
        g_train = g.iloc[tr_idx]
        g_test = g.iloc[te_idx]
        impostor_pool = df[df["user_id"] != uid]
        imp = impostor_pool.sample(n=len(g_test), random_state=random_state, replace=True)
        parts[str(uid)] = {
            "train": g_train,
            "test_genuine": g_test,
            "test_impostor": imp
        }
    return parts

def to_numpy(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    return df[feat_cols].to_numpy(dtype=np.float32)
