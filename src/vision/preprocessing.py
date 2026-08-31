from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


BBox = tuple[int, int, int, int]


def clip_bbox(bbox: BBox, frame_shape: tuple[int, ...]) -> BBox | None:
    x, y, w, h = bbox
    frame_h, frame_w = frame_shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame_w, int(x + w))
    y2 = min(frame_h, int(y + h))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def expand_bbox(
    bbox: BBox,
    margin_ratio: float,
    frame_shape: tuple[int, ...],
) -> BBox | None:
    x, y, w, h = bbox
    margin_x = int(round(w * margin_ratio))
    margin_y = int(round(h * margin_ratio))
    expanded = (
        x - margin_x,
        y - margin_y,
        w + 2 * margin_x,
        h + 2 * margin_y,
    )
    return clip_bbox(expanded, frame_shape)


def crop_roi(frame: np.ndarray, bbox: BBox | None) -> np.ndarray | None:
    if frame is None or frame.size == 0 or bbox is None:
        return None
    clipped = clip_bbox(bbox, frame.shape)
    if clipped is None:
        return None
    x, y, w, h = clipped
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return None
    return roi


def prepare_eye_for_model(
    eye_bgr: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    if eye_bgr is None or eye_bgr.size == 0:
        raise ValueError("ROI ocular inválido o vacío.")

    if eye_bgr.ndim == 2:
        rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_GRAY2RGB)
    elif eye_bgr.shape[2] == 4:
        rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)

    interpolation = (
        cv2.INTER_AREA
        if rgb.shape[1] >= width or rgb.shape[0] >= height
        else cv2.INTER_CUBIC
    )
    resized = cv2.resize(rgb, (width, height), interpolation=interpolation)
    return resized.astype(np.float32)
