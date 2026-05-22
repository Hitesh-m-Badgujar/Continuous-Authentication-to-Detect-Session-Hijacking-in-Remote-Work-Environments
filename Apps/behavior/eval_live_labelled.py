from __future__ import annotations

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
ART = BASE / "artifacts" / "realtime"

GENUINE_CSV = ART / "live_genuine.csv"
TAKEOVER_CSV = ART / "live_takeover.csv"
OUT_CSV = ART / "live_labelled_policy_metrics.csv"

def load_phase(path: Path, phase: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty")

    if "action" not in df.columns:
        raise ValueError(f"{path} missing 'action' column")

    df = df.copy()
    df["phase"] = phase
    df["action"] = df["action"].astype(str)
    return df

def main() -> None:
    genuine = load_phase(GENUINE_CSV, "genuine")
    takeover = load_phase(TAKEOVER_CSV, "takeover")

    df = pd.concat([genuine, takeover], ignore_index=True)

    # Expected behaviour:
    # genuine  -> ALLOW is correct
    # takeover -> STEP_UP or LOCK is correct
    df["correct"] = False
    df.loc[df["phase"] == "genuine", "correct"] = df["action"].eq("ALLOW")
    df.loc[df["phase"] == "takeover", "correct"] = df["action"].isin(["STEP_UP", "LOCK"])

    total_accuracy = float(df["correct"].mean())

    genuine_df = df[df["phase"] == "genuine"]
    takeover_df = df[df["phase"] == "takeover"]

    genuine_allow_rate = float(genuine_df["action"].eq("ALLOW").mean())
    genuine_false_alarm_rate = float(genuine_df["action"].isin(["STEP_UP", "LOCK"]).mean())

    takeover_detection_rate = float(takeover_df["action"].isin(["STEP_UP", "LOCK"]).mean())
    takeover_lock_rate = float(takeover_df["action"].eq("LOCK").mean())
    takeover_stepup_rate = float(takeover_df["action"].eq("STEP_UP").mean())

    summary = pd.DataFrame([
        {"metric": "live_policy_accuracy", "value": total_accuracy},
        {"metric": "genuine_allow_rate", "value": genuine_allow_rate},
        {"metric": "genuine_false_alarm_rate", "value": genuine_false_alarm_rate},
        {"metric": "takeover_detection_rate_STEP_UP_or_LOCK", "value": takeover_detection_rate},
        {"metric": "takeover_lock_rate", "value": takeover_lock_rate},
        {"metric": "takeover_stepup_rate", "value": takeover_stepup_rate},
        {"metric": "n_genuine_rows", "value": len(genuine_df)},
        {"metric": "n_takeover_rows", "value": len(takeover_df)},
        {"metric": "n_total_rows", "value": len(df)},
    ])

    summary.to_csv(OUT_CSV, index=False)

    print("\nLive labelled policy evaluation")
    print(summary.to_string(index=False))
    print(f"\nSaved -> {OUT_CSV}")

if __name__ == "__main__":
    main()