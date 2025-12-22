import cv2

from face_module.face_engine import FaceEngine


USER_ID = "user1"   # must match whatever you used in enrolment


def main():
    engine = FaceEngine()

    if USER_ID not in engine.facebank:
        print(f"ERROR: user_id '{USER_ID}' not found in facebank.")
        print("Run face_enroll.py first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face match score
        face_score = engine.match_score(frame, USER_ID)

        # Liveness score
        live_score = engine.update_liveness(frame)

        # Simple text overlay
        txt1 = f"Face score: {face_score:.2f}"
        txt2 = f"Liveness: {live_score:.2f}"

        cv2.putText(frame, txt1, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, txt2, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Face + Liveness Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
