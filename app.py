from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from src.inference.predictor import (
    EyePrediction, 
    EyeStatePredictor,
    combine_eye_predictions,
)
from src.inference.temporal_detector import PredictionSmoother, TemporalClosureDetector
from src.utils.debug_saver import DebugFrameSaver
from src.utils.helpers import (
    configure_logging,
    ensure_project_directories,
    is_probable_stream,
    load_config,
    parse_camera_source,
    project_path,
)
from src.utils.sound import SoundAlerter
from src.utils.test_logger import TestCSVLogger
from src.vision.eye_detector import EyeDetector, EyeRegions
from src.vision.face_detector import FaceDetector
from src.vision.landmark_detector import FaceLandmarkEyeDetector
from src.vision.preprocessing import BBox, crop_roi

LOGGER = logging.getLogger(__name__)


def draw_bbox(
    frame: np.ndarray,
    bbox: BBox | None,
    color: tuple[int, int, int],
    label: str = "",
) -> None:
    if bbox is None:
        return
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(
            frame,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def state_color(state: str) -> tuple[int, int, int]:
    if state == "OPEN":
        return (0, 220, 0)
    if state == "CLOSED":
        return (0, 165, 255)
    return (180, 180, 180)


def format_eye(label: str, prediction: EyePrediction | None) -> str:
    if prediction is None:
        return f"{label}: NO DETECTADO"
    return f"{label}: {prediction.state} {prediction.confidence * 100:.1f}%"


def draw_status_panel(
    frame: np.ndarray,
    left_prediction: EyePrediction | None,
    right_prediction: EyePrediction | None,
    combined: EyePrediction,
    smoothed_state: str,
    closed_duration: float,
    alert: bool,
    fps: float,
    show_fps: bool,
    roi_source: str,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (560, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        format_eye("LEFT", left_prediction),
        format_eye("RIGHT", right_prediction),
        f"STATE: {smoothed_state}",
        f"CONFIDENCE: {combined.confidence * 100:.1f}%",
        f"EYE RULE: {combined.source}",
        f"ROI SOURCE: {roi_source}",
        f"CLOSED TIME: {closed_duration:.2f} s",
    ]
    if show_fps:
        lines.append(f"FPS: {fps:.1f}")

    y = 32
    for text in lines:
        color = state_color(smoothed_state) if text.startswith("STATE:") else (255, 255, 255)
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 27

    if alert:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 210), -1)
        cv2.putText(
            frame,
            "ADVERTENCIA: CIERRE OCULAR SOSTENIDO",
            (20, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def predict_roi(
    roi: np.ndarray | None,
    predictor: EyeStatePredictor,
) -> EyePrediction | None:
    if roi is None or roi.size == 0:
        return None
    try:
        return predictor.predict(roi)
    except (ValueError, cv2.error) as exc:
        LOGGER.debug("ROI ocular inválido: %s", exc)
        return None


def open_capture(source: int | str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
    return capture


def locate_face_and_eyes(
    frame: np.ndarray,
    landmark_detector: FaceLandmarkEyeDetector,
    face_fallback: FaceDetector,
    eye_fallback: EyeDetector,
) -> tuple[BBox | None, EyeRegions, str]:
    landmark_result = landmark_detector.detect(frame)
    if landmark_result.face is not None:
        return landmark_result.face, landmark_result.eyes, landmark_result.source

    face_bbox = face_fallback.detect(frame)
    if face_bbox is None:
        return None, EyeRegions(None, None), "NONE"

    eye_regions = eye_fallback.detect(frame, face_bbox)
    return face_bbox, eye_regions, "FACE_GEOMETRY"


def main() -> None:
    configure_logging()
    config = load_config("config.yaml")
    ensure_project_directories(config)

    dataset_cfg = config["dataset"]
    inference_cfg = config["inference"]

    model_path = project_path(inference_cfg["model_path"])
    predictor = EyeStatePredictor(
        model_path=model_path,
        image_width=int(dataset_cfg["image_width"]),
        image_height=int(dataset_cfg["image_height"]),
        confidence_threshold=float(inference_cfg["confidence_threshold"]),
    )

    landmark_detector = FaceLandmarkEyeDetector(config)
    face_fallback = FaceDetector()
    eye_fallback = EyeDetector(
        margin_ratio=float(inference_cfg.get("fallback_eye_roi_margin", 0.08))
    )

    smoother = PredictionSmoother(
        history_size=int(inference_cfg["prediction_history"]),
        max_uncertain_frames=int(inference_cfg.get("max_uncertain_frames", 2)),
    )
    temporal = TemporalClosureDetector(
        threshold_seconds=float(inference_cfg["closed_eye_duration_threshold"]),
        uncertain_grace_period=float(inference_cfg["uncertain_grace_period"]),
    )
    sound = SoundAlerter(
        enabled=bool(inference_cfg["sound_alert"]),
        cooldown_seconds=float(inference_cfg["sound_cooldown_seconds"]),
    )
    debug_saver = DebugFrameSaver(config)

    source = parse_camera_source(inference_cfg["camera_source"])
    capture = open_capture(source)
    stream_like = is_probable_stream(source)
    max_failures = int(inference_cfg["max_consecutive_read_failures"])
    read_failures = 0

    test_logger = (
        TestCSVLogger()
        if bool(inference_cfg.get("enable_test_logging", False))
        else None
    )
    scenario = str(inference_cfg.get("test_scenario", "manual"))
    expected_state = str(inference_cfg.get("expected_state", ""))

    previous_tick = time.perf_counter()
    fps_ema = 0.0
    LOGGER.info("Aplicación iniciada. Presiona Q para salir.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                read_failures += 1
                if not stream_like:
                    LOGGER.info("Fin del archivo de video o frame no disponible.")
                    break
                if read_failures >= max_failures:
                    LOGGER.error(
                        "Se alcanzó el máximo de fallos consecutivos de lectura (%d).",
                        max_failures,
                    )
                    break
                time.sleep(0.05)
                continue

            read_failures = 0
            now_tick = time.perf_counter()
            dt = max(now_tick - previous_tick, 1e-6)
            previous_tick = now_tick
            instantaneous_fps = 1.0 / dt
            fps_ema = (
                instantaneous_fps
                if fps_ema == 0.0
                else 0.90 * fps_ema + 0.10 * instantaneous_fps
            )

            face_bbox, eye_regions, roi_source = locate_face_and_eyes(
                frame,
                landmark_detector,
                face_fallback,
                eye_fallback,
            )

            left_roi = crop_roi(frame, eye_regions.left)
            right_roi = crop_roi(frame, eye_regions.right)
            left_prediction = predict_roi(left_roi, predictor)
            right_prediction = predict_roi(right_roi, predictor)

            combined = combine_eye_predictions(
                left_prediction,
                right_prediction,
                allow_single_eye_fallback=bool(
                    inference_cfg.get("allow_single_eye_fallback", True)
                ),
                single_eye_min_confidence=float(
                    inference_cfg.get("single_eye_min_confidence", 0.85)
                ),
            )

            if face_bbox is None:
                raw_state = "UNCERTAIN"
            else:
                raw_state = combined.state

            smoothed_state = smoother.update(raw_state)
            temporal_status = temporal.update(smoothed_state)

            if temporal_status.alert:
                sound.trigger()

            draw_bbox(frame, face_bbox, (255, 180, 0), "FACE")
            if eye_regions.left is not None:
                draw_bbox(
                    frame,
                    eye_regions.left,
                    state_color(
                        left_prediction.state if left_prediction else "UNCERTAIN"
                    ),
                    "LEFT EYE",
                )
            if eye_regions.right is not None:
                draw_bbox(
                    frame,
                    eye_regions.right,
                    state_color(
                        right_prediction.state if right_prediction else "UNCERTAIN"
                    ),
                    "RIGHT EYE",
                )

            draw_status_panel(
                frame=frame,
                left_prediction=left_prediction,
                right_prediction=right_prediction,
                combined=combined,
                smoothed_state=smoothed_state,
                closed_duration=temporal_status.closed_duration,
                alert=temporal_status.alert,
                fps=fps_ema,
                show_fps=bool(inference_cfg["show_fps"]),
                roi_source=roi_source,
            )

            debug_reason = debug_saver.choose_reason(
                combined.state,
                combined.confidence,
            )
            debug_saver.maybe_save(
                frame=frame,
                left_roi=left_roi,
                right_roi=right_roi,
                reason=debug_reason,
                state=combined.state,
                confidence=combined.confidence,
                roi_source=roi_source,
            )

            if test_logger is not None:
                test_logger.log(
                    scenario=scenario,
                    camera_source=source,
                    expected_state=expected_state,
                    predicted_state=smoothed_state,
                    confidence=combined.confidence,
                    closed_duration=temporal_status.closed_duration,
                    alert=temporal_status.alert,
                    notes=f"roi_source={roi_source}; eye_rule={combined.source}",
                )

            cv2.imshow("TTR - Deteccion de senales visuales", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    except KeyboardInterrupt:
        LOGGER.info("Interrupción por teclado.")
    finally:
        capture.release()
        landmark_detector.close()
        cv2.destroyAllWindows()
        LOGGER.info("Aplicación finalizada.")


if __name__ == "__main__":
    main()
