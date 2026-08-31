from __future__ import annotations

import cv2
import numpy as np

from src.vision.preprocessing import BBox


class FaceDetector:
    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (80, 80),
    ) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"No se pudo cargar el detector facial: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame_bgr: np.ndarray) -> BBox | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        if len(faces) == 0:
            return None

        # Se usa el rostro más grande como sujeto principal.
        x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        return int(x), int(y), int(w), int(h)
