# Apps/behavior/face_runtime.py
"""
Simple face embedding + heuristic liveness for the 3F trust engine.

- Enrol: store a single face template vector.
- Score: compute face_match (cosine similarity) + liveness in [0, 1].

Liveness is improved vs the naive version by:
  - Focusing on the face ROI only
  - Comparing face motion vs global frame motion (replay detection heuristic)
  - Tracking motion of the face bounding box over time
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "Models" / "face"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HAAR_PATH = str(BASE_DIR / "Models" / "haarcascade_frontalface_default.xml")


def _load_cascade() -> cv2.CascadeClassifier:
    casc = cv2.CascadeClassifier(HAAR_PATH)
    if casc.empty():
        # Fallback to OpenCV's default path if your file is missing
        casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return casc


def _b64_to_image(image_b64: str) -> Optional[np.ndarray]:
    """
    Decode a data:image/jpeg;base64,... or raw base64 string to BGR image.
    """
    if not image_b64:
        return None
    # Strip possible "data:image/jpeg;base64," prefix
    if image_b64.startswith("data:"):
        try:
            image_b64 = image_b64.split(",", 1)[1]
        except Exception:
            pass
    try:
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _unit_vector(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(x) + 1e-8
    return x / n


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = _unit_vector(a)
    b = _unit_vector(b)
    return float(np.dot(a, b))


@dataclass
class FaceEngine:
    """
    Runtime face engine with a bit of temporal memory for liveness.

    Template is stored in Models/face/template_simple.npz.
    """

    cascade: cv2.CascadeClassifier = field(default_factory=_load_cascade)
    template_vec: Optional[np.ndarray] = None
    enrolled: bool = False

    # Temporal memory for liveness
    last_frame_gray: Optional[np.ndarray] = None
    last_face_roi_gray: Optional[np.ndarray] = None
    bbox_history: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._load_template()

    # ------------------------------------------------------------------
    # Template persistence
    # ------------------------------------------------------------------

    @property
    def template_path(self) -> Path:
        return MODELS_DIR / "template_simple.npz"

    def _load_template(self) -> None:
        if self.template_path.is_file():
            try:
                data = np.load(self.template_path)
                vec = data.get("template")
                if vec is not None and vec.size > 0:
                    self.template_vec = _unit_vector(vec)
                    self.enrolled = True
                    return
            except Exception:
                pass
        self.template_vec = None
        self.enrolled = False

    def _save_template(self, vec: np.ndarray) -> None:
        vec = _unit_vector(vec)
        np.savez_compressed(self.template_path, template=vec)
        self.template_vec = vec
        self.enrolled = True

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _detect_face(self, gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Return (x, y, w, h) of the largest detected face, or None.
        """
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if faces is None or len(faces) == 0:
            return None
        # Choose largest box by area
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = map(int, faces[0])
        return x, y, w, h

    def _extract_face_roi(self, gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        h_img, w_img = gray.shape[:2]
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w_img, x + w)
        y1 = min(h_img, y + h)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            # Fallback: whole frame
            roi = gray
        return roi

    def _embed_face(self, roi_gray: np.ndarray) -> np.ndarray:
        """
        Extremely simple embedding: resize to 64x64 grayscale, flatten, L2-normalise.
        """
        roi_resized = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        vec = roi_resized.astype(np.float32).reshape(-1)
        return _unit_vector(vec)

    # ------------------------------------------------------------------
    # Liveness heuristic
    # ------------------------------------------------------------------

    def _compute_liveness(
        self,
        frame_gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
        roi_gray: np.ndarray,
    ) -> float:
        """
        Heuristic liveness in [0, 1].

        Components:
          - face ROI pixel motion vs last face ROI
          - face bounding box motion vs last bbox
          - relative motion: face vs global frame

        Idea:
          - If everything moves together (screen replay) -> relative motion small.
          - If face moves/changes more than background -> higher liveness.
        """
        h, w = frame_gray.shape[:2]  # noqa: F841  (kept for clarity)

        # First frame: no temporal info yet
        if self.last_frame_gray is None or self.last_face_roi_gray is None or not self.bbox_history:
            self.last_frame_gray = frame_gray.copy()
            self.last_face_roi_gray = roi_gray.copy()
            self.bbox_history = [bbox]
            # Low-ish default: neither clearly fake nor clearly live
            return 0.3

        # --- Face ROI motion (local) ---
        cur_roi_small = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        prev_roi_small = cv2.resize(self.last_face_roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi_diff = np.mean(
            np.abs(cur_roi_small.astype(np.float32) - prev_roi_small.astype(np.float32))
        ) / 255.0
        roi_diff = float(np.clip(roi_diff, 0.0, 1.0))

        # --- Global frame motion (background) ---
        cur_frame_small = cv2.resize(frame_gray, (128, 72), interpolation=cv2.INTER_AREA)
        prev_frame_small = cv2.resize(self.last_frame_gray, (128, 72), interpolation=cv2.INTER_AREA)
        global_diff = np.mean(
            np.abs(cur_frame_small.astype(np.float32) - prev_frame_small.astype(np.float32))
        ) / 255.0
        global_diff = float(np.clip(global_diff, 0.0, 1.0))

        # Relative motion: how much more the face changes than the rest
        rel_motion = roi_diff - global_diff
        # Boost positive differences; clamp to [0, 1]
        rel_motion = float(np.clip(rel_motion * 3.0, 0.0, 1.0))

        # --- Bounding box motion (coarse head movement) ---
        prev_bbox = self.bbox_history[-1]
        x, y, w_box, h_box = bbox
        px, py, pw, ph = prev_bbox

        cx, cy = x + w_box / 2.0, y + h_box / 2.0
        pcx, pcy = px + pw / 2.0, py + ph / 2.0

        dist = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
        diag = np.sqrt(frame_gray.shape[1] ** 2 + frame_gray.shape[0] ** 2) + 1e-8
        # Normalise by quarter of frame diagonal ≈ "reasonable" motion scaling
        bbox_motion = float(np.clip(dist / (0.25 * diag), 0.0, 1.0))

        # Combine components
        w_rel = 0.5   # face vs background
        w_roi = 0.2   # local pixel changes
        w_bbox = 0.3  # head/body movement

        liveness_raw = w_rel * rel_motion + w_roi * roi_diff + w_bbox * bbox_motion
        liveness = float(np.clip(liveness_raw, 0.0, 1.0))

        # Update temporal memory
        self.last_frame_gray = frame_gray.copy()
        self.last_face_roi_gray = roi_gray.copy()
        self.bbox_history.append(bbox)
        if len(self.bbox_history) > 10:
            self.bbox_history.pop(0)

        return liveness

    # ------------------------------------------------------------------
    # Public API used by Django views
    # ------------------------------------------------------------------

    def enroll_from_b64(self, image_b64: str) -> Dict[str, Any]:
        """
        Enrol a new template from a single frame.
        """
        img = _b64_to_image(image_b64)
        if img is None:
            return {"ok": False, "error": "decode_failed"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bbox = self._detect_face(gray)
        if bbox is None:
            return {"ok": False, "error": "no_face"}

        roi = self._extract_face_roi(gray, bbox)
        vec = self._embed_face(roi)
        self._save_template(vec)

        # Reset temporal memory after enrol
        self.last_frame_gray = gray.copy()
        self.last_face_roi_gray = roi.copy()
        self.bbox_history = [bbox]

        return {"ok": True, "enrolled": True}

    def score_from_b64(self, image_b64: str) -> Dict[str, Any]:
        """
        Compute face_match + liveness from a frame.

        If not enrolled, we DO NOT return scores (front-end will treat as N/A).
        """
        if not self.enrolled or self.template_vec is None:
            return {"ok": True, "face_match": None, "liveness": None, "note": "not_enrolled"}

        img = _b64_to_image(image_b64)
        if img is None:
            return {"ok": False, "error": "decode_failed", "face_match": None, "liveness": None}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bbox = self._detect_face(gray)
        if bbox is None:
            return {"ok": False, "error": "no_face", "face_match": None, "liveness": None}

        roi = self._extract_face_roi(gray, bbox)
        vec = self._embed_face(roi)

        # Cosine similarity [-1, 1] -> [0, 1]
        sim = _cosine_sim(self.template_vec, vec)
        sim01 = float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))

        # Improved liveness heuristic
        live = self._compute_liveness(gray, bbox, roi)

        return {
            "ok": True,
            "face_match": sim01,
            "liveness": live,
        }

    def health(self) -> Dict[str, Any]:
        """
        Optional for dashboard / debug.
        """
        return {
            "ok": True,
            "enrolled": bool(self.enrolled),
            "has_template": self.template_vec is not None,
            "template_path": str(self.template_path),
        }
