# Apps/behavior/trust_logging.py
import os
import csv

LOG_CSV = os.path.join("Data", "live_trust_timeseries.csv")

os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

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
