import os
import json
import cv2
import numpy as np
from collections import deque

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

import mediapipe as mp

from face_module.crypto_utils import get_fernet, encrypt_bytes, decrypt_bytes


# ---- Paths ----

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)  # project root (where manage.py lives)
MODELS_DIR = os.path.join(BASE_DIR, "Models")
FACE_DIR = os.path.join(MODELS_DIR, "face")

os.makedirs(FACE_DIR, exist_ok=True)

# Encrypted facebank path (main storage)
FACEBANK_ENC_PATH = os.path.join(FACE_DIR, "facebank.enc")
# Legacy plaintext JSON path (for backward-compat migration)
FACEBANK_JSON_PATH = os.path.join(FACE_DIR, "facebank.json")


# ---- Small helpers ----

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _deserialize_facebank(data: str) -> dict:
    raw = json.loads(data)
    return {k: np.array(v, dtype=np.float32) for k, v in raw.items()}


def _serialize_facebank(facebank: dict) -> str:
    serialisable = {k: v.tolist() for k, v in facebank.items()}
    return json.dumps(serialisable, indent=2)


def load_facebank() -> dict:
    """
    Load facebank from encrypted .enc file if it exists; otherwise fall back
    to legacy plaintext JSON and migrate it to encrypted form.
    """
    fernet = get_fernet()

    # 1) Preferred: encrypted file
    if os.path.exists(FACEBANK_ENC_PATH):
        try:
            with open(FACEBANK_ENC_PATH, "rb") as f:
                token = f.read()
            decrypted = decrypt_bytes(token, fernet).decode("utf-8")
            return _deserialize_facebank(decrypted)
        except Exception as e:  # noqa: BLE001
            print("[WARN] Failed to decrypt facebank.enc:", e)
            # fall through to JSON as a last resort

    # 2) Legacy: plaintext JSON (migrate if present)
    if os.path.exists(FACEBANK_JSON_PATH):
        try:
            with open(FACEBANK_JSON_PATH, "r", encoding="utf-8") as f:
                data = f.read()
            facebank = _deserialize_facebank(data)
            # migrate to encrypted file
            save_facebank(facebank)
            return facebank
        except Exception as e:  # noqa: BLE001
            print("[WARN] Failed to load legacy facebank.json:", e)
            return {}

    # 3) Nothing found
    return {}


def save_facebank(facebank: dict) -> None:
    """
    Serialize and encrypt the facebank to FACEBANK_ENC_PATH.
    """
    fernet = get_fernet()
    data = _serialize_facebank(facebank).encode("utf-8")
    token = encrypt_bytes(data, fernet)
    with open(FACEBANK_ENC_PATH, "wb") as f:
        f.write(token)


# ---- FaceEngine ----

class FaceEngine:
    """
    Handles:
      - Face detection & alignment (MTCNN)
      - Face embeddings (InceptionResnetV1 pretrained on VGGFace2)
      - Passive liveness using MediaPipe FaceMesh:
          * eye-blink (eye openness variation)
          * mouth motion
          * small head/nose motion
      - Encrypted storage of facial templates (facebank) using Fernet.
    """

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Face detector / aligner (MTCNN)
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            keep_all=False,
            device=self.device
        )

        # Face embedding model (VGGFace2 pre-trained Facenet-style)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # Facebank: user_id -> embedding (numpy array), loaded from encrypted store
        self.facebank = load_facebank()

        # Mediapipe FaceMesh for liveness
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Histories for liveness
        self.nose_history = deque(maxlen=15)    # head motion
        self.eye_history = deque(maxlen=30)     # eye openness over time
        self.mouth_history = deque(maxlen=30)   # mouth openness over time

    # ---- Face embeddings / verification ----

    def embed_frame(self, frame_bgr: np.ndarray):
        """
        Takes a BGR frame from OpenCV, returns a 512-D embedding (numpy) or None if no face.
        Uses MTCNN for detection + alignment, then InceptionResnetV1 for embeddings.
        """
        # Convert BGR -> RGB for MTCNN
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            face_tensor = self.mtcnn(rgb)  # cropped + aligned face
        if face_tensor is None:
            return None
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)
        face_tensor = face_tensor.to(self.device)
        with torch.no_grad():
            emb = self.model(face_tensor).cpu().numpy()[0]
        return emb.astype(np.float32)

    def enroll_from_frames(self, user_id: str, frames):
        """
        Given a list of BGR frames, compute average embedding and store in facebank.
        Returns: number of frames used (float) for logging.
        """
        embs = []
        for f in frames:
            emb = self.embed_frame(f)
            if emb is not None:
                embs.append(emb)
        if not embs:
            return 0.0
        embs = np.stack(embs, axis=0)
        mean_emb = embs.mean(axis=0)
        self.facebank[user_id] = mean_emb
        save_facebank(self.facebank)
        return float(len(embs))

    def match_score(self, frame_bgr: np.ndarray, user_id: str) -> float:
        """
        Compute face match score [0,1] between current frame and stored template for user_id.
        If no template or no face detected, returns 0.
        """
        if user_id not in self.facebank:
            return 0.0
        template = self.facebank[user_id]
        emb = self.embed_frame(frame_bgr)
        if emb is None:
            return 0.0
        cos = cosine_similarity(emb, template)  # in [-1, 1]
        score = (cos + 1.0) / 2.0  # map to [0,1]
        score = max(0.0, min(1.0, score))
        return score

    # ---- Liveness: eye-blink + mouth motion + head motion ----

    def _update_liveness_features(self, frame_bgr: np.ndarray) -> None:
        """
        Extract eye and mouth openness metrics from MediaPipe FaceMesh
        and update history deques. Also track nose motion.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            # no face → reset histories
            self.eye_history.clear()
            self.mouth_history.clear()
            self.nose_history.clear()
            return

        face_landmarks = results.multi_face_landmarks[0].landmark

        # Landmark indices from MediaPipe FaceMesh:
        # Eye centres
        L_EYE_CENTER, R_EYE_CENTER = 33, 263
        # Eye top/bottom
        L_EYE_TOP, L_EYE_BOT = 159, 145
        R_EYE_TOP, R_EYE_BOT = 386, 374
        # Mouth corners + lips
        MOUTH_LEFT, MOUTH_RIGHT = 61, 291
        LIP_UP, LIP_DOWN = 13, 14

        def euclidean(p1, p2):
            return np.linalg.norm(
                np.array([p1.x, p1.y], dtype=np.float32)
                - np.array([p2.x, p2.y], dtype=np.float32)
            )

        # Normalisation scale: distance between eye centres
        io = euclidean(face_landmarks[L_EYE_CENTER], face_landmarks[R_EYE_CENTER])
        if io < 1e-6:
            return

        # Eye openness (average of left/right)
        eye_open_L = euclidean(face_landmarks[L_EYE_TOP], face_landmarks[L_EYE_BOT]) / io
        eye_open_R = euclidean(face_landmarks[R_EYE_TOP], face_landmarks[R_EYE_BOT]) / io
        eye_open = 0.5 * (eye_open_L + eye_open_R)

        # Mouth openness (vertical gap normalised)
        mouth_open = euclidean(face_landmarks[LIP_UP], face_landmarks[LIP_DOWN]) / io

        self.eye_history.append(float(eye_open))
        self.mouth_history.append(float(mouth_open))

        # Nose / head motion
        h, w, _ = frame_bgr.shape
        nose = face_landmarks[1]  # approximate nose tip
        x, y = int(nose.x * w), int(nose.y * h)
        self.nose_history.append((x, y))

    def liveness_score(self) -> float:
        """
        Compute liveness score [0,1] based on:
          - variation in eye openness (blinks)
          - variation in mouth openness (speech/mouth motion)
          - small contribution from nose/head motion
        """
        # Need some history
        if len(self.eye_history) < 5 or len(self.mouth_history) < 5:
            return 0.0

        eye_vals = np.array(self.eye_history, dtype=np.float32)
        mouth_vals = np.array(self.mouth_history, dtype=np.float32)

        # Variation (std deviation) over the window
        eye_var = float(np.std(eye_vals))
        mouth_var = float(np.std(mouth_vals))

        # Nose motion magnitude
        nose_motion = 0.0
        if len(self.nose_history) >= 2:
            pts = np.array(self.nose_history, dtype=np.float32)
            diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            nose_motion = float(np.mean(diffs))

        # Heuristic scaling:
        # - eye_var ~0.002–0.01 when blinking
        # - mouth_var ~0.002–0.02 when talking
        # - nose_motion ~0–3 pixels with small head movements
        eye_score = min(1.0, eye_var / 0.01)
        mouth_score = min(1.0, mouth_var / 0.02)
        nose_score = min(1.0, nose_motion / 2.0)

        # Combine with weights; favour eyes and mouth
        score = 0.5 * eye_score + 0.3 * mouth_score + 0.2 * nose_score
        score = max(0.0, min(1.0, score))
        return score

    def update_liveness(self, frame_bgr: np.ndarray) -> float:
        """
        Call this every frame; updates internal histories and returns current liveness score.
        """
        self._update_liveness_features(frame_bgr)
        return self.liveness_score()
