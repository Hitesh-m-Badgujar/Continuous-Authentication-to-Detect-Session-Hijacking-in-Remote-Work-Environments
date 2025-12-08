import json
import time

import cv2
import requests

from face_module.face_engine import FaceEngine

# Adjust if you enrolled under a different ID
USER_ID = "user1"

# Your Django fusion endpoint
FUSION_URL = "http://127.0.0.1:8000/behavior/fuse_scores"


def main():
    engine = FaceEngine()

    if USER_ID not in engine.facebank:
        print(f"[ERROR] user_id '{USER_ID}' not found in facebank. Run face_enroll.py first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Streaming face + liveness to fusion endpoint.")
    print("[INFO] Press 'q' to quit.")

    last_post_time = 0.0
    post_interval = 0.3  # seconds between HTTP posts to avoid spamming

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face match and liveness scores
        face_score = engine.match_score(frame, USER_ID)
        live_score = engine.update_liveness(frame)

        overall_trust = None
        action = None

        now = time.time()
        if now - last_post_time >= post_interval:
            last_post_time = now

            payload = {
                "user_id": USER_ID,
                # For now we only send facial signals; behavioural can be added later.
                "kb_trust": None,
                "mouse_trust": None,
                "face_match": face_score,
                "liveness": live_score,
            }

            try:
                resp = requests.post(FUSION_URL, json=payload, timeout=1.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        overall_trust = data.get("overall_trust")
                        action = data.get("action")
                    else:
                        print("[WARN] Fusion error:", data)
                else:
                    print(f"[WARN] Fusion HTTP {resp.status_code}: {resp.text}")
            except Exception as e:
                print("[WARN] Request to fusion endpoint failed:", e)

        # Overlay scores on the frame
        txt_face = f"Face: {face_score:.2f}"
        txt_live = f"Live: {live_score:.2f}"

        cv2.putText(frame, txt_face, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, txt_live, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if overall_trust is not None and action is not None:
            txt_trust = f"Trust*: {overall_trust:.2f} ({action})"
            cv2.putText(frame, txt_trust, (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imshow("Face → Fusion (backend)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
