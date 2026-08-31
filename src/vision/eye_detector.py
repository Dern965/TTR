from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.vision.preprocessing import BBox, clip_bbox, expand_bbox


@dataclass(frozen=True)
class EyeRegions:
    left: BBox | None
    right: BBox | None


class EyeDetector:
    """Fallback geométrico para los ojos a partir del bounding box del rostro.

    Ya no usa Haar para detectar la apariencia de un ojo. Si MediaPipe pierde
    temporalmente los landmarks pero el rostro sigue visible, estas regiones
    aproximadas permiten continuar enviando ROIs al clasificador.
    """

    def __init__(self, margin_ratio: float = 0.08) -> None:
        self.margin_ratio = float(margin_ratio)

    def detect(self, frame_bgr: np.ndarray, face_bbox: BBox | None) -> EyeRegions:
        if frame_bgr is None or frame_bgr.size == 0 or face_bbox is None:
            return EyeRegions(None, None)

        clipped_face = clip_bbox(face_bbox, frame_bgr.shape)
        if clipped_face is None:
            return EyeRegions(None, None)

        fx, fy, fw, fh = clipped_face

        # Regiones amplias que cubren párpados completos, aun estando cerrados.
        left_raw = (
            int(round(fx + 0.08 * fw)),
            int(round(fy + 0.22 * fh)),
            int(round(0.39 * fw)),
            int(round(0.27 * fh)),
        )
        right_raw = (
            int(round(fx + 0.53 * fw)),
            int(round(fy + 0.22 * fh)),
            int(round(0.39 * fw)),
            int(round(0.27 * fh)),
        )

        left = expand_bbox(left_raw, self.margin_ratio, frame_bgr.shape)
        right = expand_bbox(right_raw, self.margin_ratio, frame_bgr.shape)
        return EyeRegions(left, right)
