from pathlib import Path
import json, numpy as np, pandas as pd
from Apps.behavior.ae_conditional import load_runtime, residuals

MODELS_DIR = Path("Models/cae_kb")
DATA_PRIMARY = Path("Data/kb_windows_clean.csv")
DATA_FALLBACK = Path("Data/kb_windows.csv")
MIN_PER_USER = 40

FEATURES = [
    "dwell_mean","dwell_std","dwell_p10","dwell_p50","dwell_p90",
    "dd_mean","dd_std","dd_p10","dd_p50","dd_p90",
    "ud_mean","ud_std","ud_p10","ud_p50","ud_p90",
    "backspace_rate","burst_mean","idle_frac"
]

def load_data():
    path = DATA_PRIMARY if DATA_PRIMARY.exists() else DATA_FALLBACK
    df = pd.read_csv(path).dropna(subset=FEATURES).copy()
    df["user_id"] = df["user_id"].astype(str)
    vc = df["user_id"].value_counts()
    keep = vc[vc >= MIN_PER_USER].index
    df = df[df["user_id"].isin(keep)].copy()
    return df

def main():
    scaler, model, features, th_json, user_to_idx = load_runtime(MODELS_DIR)
    df = load_data()
    if df.empty:
        raise SystemExit("No data for calibration.")

    thresholds = {}
    for u in sorted(df["user_id"].unique()):
        g = df[df.user_id == u]
        imp = df[df.user_id != u]
        if len(imp) > 5000:
            imp = imp.sample(5000, random_state=7)

        Xg = g[FEATURES].to_numpy(np.float32)
        Xi = imp[FEATURES].to_numpy(np.float32)
        uid_g = np.full(len(Xg), user_to_idx[u], np.int32)
        uid_i = np.full(len(Xi), user_to_idx[u], np.int32)

        rg = residuals(model, scaler, Xg, uid_g)
        ri = residuals(model, scaler, Xi, uid_i)
        cand = np.quantile(np.concatenate([rg, ri]), np.linspace(0.05,0.95,19))

        best = None
        for tau in cand:
            FAR = float((ri < tau).mean())
            FRR = float((rg >= tau).mean())
            err = 0.5*(FAR+FRR)
            if best is None or err < best[0]:
                best = (err, float(tau), FAR, FRR, len(rg), len(ri))
        err, tau, FAR, FRR, ng, ni = best
        thresholds[u] = tau
        print(f"{u}\tbest_tau={tau:.6f}\tFAR={FAR:.3f}\tFRR={FRR:.3f}\terr={err:.3f}\t n_g={ng}\t n_i={ni}")

    th_json["policy"] = "per-user-balanced"
    th_json["values"] = thresholds
    (MODELS_DIR / "thresholds.json").write_text(json.dumps(th_json, indent=2))
    print(f"\nWROTE {MODELS_DIR/'thresholds.json'} with {len(thresholds)} users calibrated.")

if __name__ == "__main__":
    main()
