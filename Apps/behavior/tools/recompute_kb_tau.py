# Apps/behavior/tools/recompute_kb_tau.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from Apps.behavior.ae_conditional import RuntimeScorer, FEATURE_COLS, DEFAULT_WINDOWS_CSV, MODELS_DIR

OUT_META = MODELS_DIR / "cae_kb" / "kb_meta.json"

def main():
    sc = RuntimeScorer()  # loads same scaler+model runtime uses
    df = pd.read_csv(DEFAULT_WINDOWS_CSV)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    res = sc._compute_residuals(X, sc.scaler, sc.model)  # same function as runtime
    tau95 = float(np.percentile(res, 95))
    meta = {"tau": tau95, "res_metric": "same_as_runtime", "feature_cols": FEATURE_COLS}
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"Tau@95 = {tau95:.6f} → wrote {OUT_META}")

if __name__ == "__main__":
    main()
