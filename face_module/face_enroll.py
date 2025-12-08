import cv2

from face_module.face_engine import FaceEngine


USER_ID = "user1"          # change this if you like
NUM_FRAMES = 30            # how many frames to use for enrolment


def main():
    engine = FaceEngine()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    collected_frames = []
    print("Press 'e' to start collecting frames for enrolment.")
    print("Press 'q' to quit without enrolment.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, "Press 'e' to ENROL, 'q' to quit",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Face Enrolment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            # Collect NUM_FRAMES frames while you stay in front of camera
            print("Collecting frames... stay still and look at the camera.")
            collected_frames = []
            count = 0
            while count < NUM_FRAMES:
                ret2, frame2 = cap.read()
                if not ret2:
                    break
                collected_frames.append(frame2.copy())
                count += 1
                cv2.putText(frame2, f"Collecting {count}/{NUM_FRAMES}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Face Enrolment", frame2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            used = engine.enroll_from_frames(USER_ID, collected_frames)
            print(f"Enrolment complete. Used {used} frames with detected face.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
