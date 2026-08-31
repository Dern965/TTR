from __future__ import annotations

import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.utils.helpers import project_path
from src.vision.eye_detector import EyeRegions
from src.vision.preprocessing import BBox, clip_bbox

LOGGER = logging.getLogger(__name__)

# Contornos oculares de MediaPipe Face Landmarker.
# Los nombres anatómicos no se usan para rotular la interfaz: al final se ordenan
# por coordenada X para mostrar LEFT/RIGHT desde la perspectiva de la imagen.
RIGHT_EYE_LANDMARKS = (
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
)
LEFT_EYE_LANDMARKS = (
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398,
)


@dataclass(frozen=True)
class LandmarkDetection:
    face: BBox | None
    eyes: EyeRegions
    source: str
    landmarks_detected: bool


class FaceLandmarkEyeDetector:
    """Localiza rostro y regiones oculares con MediaPipe Face Landmarker.

    La localización de los ojos se basa en puntos faciales, no en detectar la
    apariencia de un "ojo abierto". Esto permite conservar una ROI cuando el
    párpado se cierra y hace el recorte más tolerante a lentes.
    """

    def __init__(self, config: dict) -> None:
        landmark_cfg = config["landmarks"]
        self.model_path = project_path(landmark_cfg["model_path"])
        self.model_url = str(landmark_cfg["model_url"])
        self.auto_download_model = bool(landmark_cfg.get("auto_download_model", True))
        self.face_margin = float(landmark_cfg.get("face_margin", 0.05))
        self.eye_margin = float(landmark_cfg.get("eye_roi_margin", 0.30))
        self.min_eye_height_ratio = float(
            landmark_cfg.get("min_eye_height_ratio_to_width", 0.48)
        )
        self.roi_ema_alpha = float(landmark_cfg.get("roi_ema_alpha", 0.65))
        self.roi_hold_frames = int(landmark_cfg.get("roi_hold_frames", 4))

        self._last_timestamp_ms = 0
        self._missed_frames = 0
        self._last_face: BBox | None = None
        self._last_left: BBox | None = None
        self._last_right: BBox | None = None

        self._ensure_model()
        self._mp = self._import_mediapipe()
        self._landmarker = self._create_landmarker(landmark_cfg)

    @staticmethod
    def _import_mediapipe():
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe no está instalado. Ejecuta: python -m pip install -r requirements.txt"
            ) from exc
        return mp

    def _ensure_model(self) -> None:
        if self.model_path.exists() and self.model_path.stat().st_size > 100_000:
            return

        if not self.auto_download_model:
            raise FileNotFoundError(
                f"No existe el modelo de landmarks: {self.model_path}. "
                "Activa landmarks.auto_download_model o coloca el archivo manualmente."
            )

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.model_path.with_suffix(self.model_path.suffix + ".download")
        LOGGER.info("Descargando modelo MediaPipe Face Landmarker...")
        LOGGER.info("Destino: %s", self.model_path)

        try:
            urllib.request.urlretrieve(self.model_url, temp_path)
            if not temp_path.exists() or temp_path.stat().st_size <= 100_000:
                raise RuntimeError("La descarga del modelo de landmarks parece incompleta.")
            os.replace(temp_path, self.model_path)
        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "No se pudo obtener face_landmarker.task. Revisa tu conexión o descarga "
                f"manualmente el modelo indicado en config.yaml. Detalle: {exc}"
            ) from exc

    def _create_landmarker(self, landmark_cfg: dict):
        mp = self._mp
        try:
            base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=float(
                    landmark_cfg.get("min_face_detection_confidence", 0.50)
                ),
                min_face_presence_confidence=float(
                    landmark_cfg.get("min_face_presence_confidence", 0.50)
                ),
                min_tracking_confidence=float(
                    landmark_cfg.get("min_tracking_confidence", 0.50)
                ),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            return mp.tasks.vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo inicializar MediaPipe Face Landmarker. "
                f"Modelo: {self.model_path}. Detalle: {exc}"
            ) from exc

    def close(self) -> None:
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray) -> LandmarkDetection:
        if frame_bgr is None or frame_bgr.size == 0:
            return self._cached_or_empty()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb,
        )

        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        try:
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            LOGGER.debug("MediaPipe no pudo procesar el frame: %s", exc)
            return self._cached_or_empty()

        if not result.face_landmarks:
            return self._cached_or_empty()

        landmarks = result.face_landmarks[0]
        frame_h, frame_w = frame_bgr.shape[:2]

        face_box = self._bbox_from_indices(
            landmarks,
            range(len(landmarks)),
            frame_w,
            frame_h,
            margin_ratio=self.face_margin,
        )
        eye_a = self._eye_bbox(
            landmarks,
            RIGHT_EYE_LANDMARKS,
            frame_w,
            frame_h,
        )
        eye_b = self._eye_bbox(
            landmarks,
            LEFT_EYE_LANDMARKS,
            frame_w,
            frame_h,
        )

        if face_box is None or eye_a is None or eye_b is None:
            return self._cached_or_empty()

        # LEFT/RIGHT corresponden a izquierda/derecha de la imagen mostrada.
        ordered = sorted((eye_a, eye_b), key=lambda box: box[0] + box[2] / 2.0)
        left_box, right_box = ordered[0], ordered[1]

        face_box = self._smooth_bbox(self._last_face, face_box)
        left_box = self._smooth_bbox(self._last_left, left_box)
        right_box = self._smooth_bbox(self._last_right, right_box)

        self._last_face = face_box
        self._last_left = left_box
        self._last_right = right_box
        self._missed_frames = 0

        return LandmarkDetection(
            face=face_box,
            eyes=EyeRegions(left_box, right_box),
            source="LANDMARKS",
            landmarks_detected=True,
        )

    def _cached_or_empty(self) -> LandmarkDetection:
        self._missed_frames += 1
        if (
            self._missed_frames <= self.roi_hold_frames
            and self._last_face is not None
            and self._last_left is not None
            and self._last_right is not None
        ):
            return LandmarkDetection(
                face=self._last_face,
                eyes=EyeRegions(self._last_left, self._last_right),
                source="CACHE",
                landmarks_detected=False,
            )

        return LandmarkDetection(
            face=None,
            eyes=EyeRegions(None, None),
            source="NONE",
            landmarks_detected=False,
        )

    def _eye_bbox(
        self,
        landmarks,
        indices: Iterable[int],
        frame_w: int,
        frame_h: int,
    ) -> BBox | None:
        raw = self._bbox_from_indices(
            landmarks,
            indices,
            frame_w,
            frame_h,
            margin_ratio=self.eye_margin,
        )
        if raw is None:
            return None

        x, y, w, h = raw
        min_h = max(h, int(round(w * self.min_eye_height_ratio)))
        center_y = y + h / 2.0
        adjusted = (x, int(round(center_y - min_h / 2.0)), w, min_h)
        return clip_bbox(adjusted, (frame_h, frame_w, 3))

    @staticmethod
    def _bbox_from_indices(
        landmarks,
        indices: Iterable[int],
        frame_w: int,
        frame_h: int,
        margin_ratio: float,
    ) -> BBox | None:
        points: list[tuple[float, float]] = []
        for idx in indices:
            if idx >= len(landmarks):
                continue
            landmark = landmarks[idx]
            x = float(landmark.x) * frame_w
            y = float(landmark.y) * frame_h
            if np.isfinite(x) and np.isfinite(y):
                points.append((x, y))

        if not points:
            return None

        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())

        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        margin_x = width * margin_ratio
        margin_y = height * margin_ratio

        candidate = (
            int(round(x1 - margin_x)),
            int(round(y1 - margin_y)),
            int(round(width + 2.0 * margin_x)),
            int(round(height + 2.0 * margin_y)),
        )
        return clip_bbox(candidate, (frame_h, frame_w, 3))

    def _smooth_bbox(self, previous: BBox | None, current: BBox) -> BBox:
        if previous is None:
            return current
        alpha = float(np.clip(self.roi_ema_alpha, 0.0, 1.0))
        values = [
            int(round(alpha * cur + (1.0 - alpha) * prev))
            for prev, cur in zip(previous, current)
        ]
        return tuple(values)  # type: ignore[return-value]
