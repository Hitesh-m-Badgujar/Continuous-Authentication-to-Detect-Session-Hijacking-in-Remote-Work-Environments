# Apps/behavior/trust_logging.py
import os
import csv
from pathlib import Path

# Log live trust for the demo in a predictable location that also exists in the
# submitted zip: H1/artifacts/realtime/live_trust_timeseries.csv
BASE_DIR = Path(__file__).resolve().parents[2]  # .../H1
LOG_PATH = BASE_DIR / "artifacts" / "realtime" / "live_trust_timeseries.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

LOG_CSV = str(LOG_PATH)

_HEADER = [
    "session_id",
    "t_ms",
    "label",
    "kb_trust",
    "mouse_trust",
    "behavioural_trust",
    "face_trust",
    "fused_trust",
    "action",
]

def append_trust_row(
    session_id,
    t_ms,
    label,
    kb_trust,
    mouse_trust,
    behavioural_trust,
    face_trust,
    fused_trust,
    action,
):
    file_exists = os.path.exists(LOG_CSV)

    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_HEADER)
        writer.writerow([
            session_id,
            int(t_ms),
            int(label),
            float(kb_trust),
            float(mouse_trust),
            float(behavioural_trust),
            float(face_trust),
            float(fused_trust),
            action,
        ])
