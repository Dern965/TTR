from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from src.utils.helpers import project_path
from src.vision.preprocessing import prepare_eye_for_model


@dataclass(frozen=True)
class EyePrediction:
    state: str
    confidence: float
    open_probability: float
    source: str = "BOTH_EYES"


class EyeStatePredictor:
    def __init__(
        self,
        model_path: str | Path,
        image_width: int,
        image_height: int,
        confidence_threshold: float = 0.60,
    ) -> None:
        self.model_path = project_path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"No existe el modelo: {self.model_path}")

        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow no está instalado. Ejecuta: python -m pip install -r requirements.txt"
            ) from exc
        self.model = tf.keras.models.load_model(self.model_path)
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.confidence_threshold = float(confidence_threshold)
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> list[str]:
        metadata_path = self.model_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = json.load(file)
            names = metadata.get("class_names")
            if names == ["closed", "open"]:
                return names
        return ["closed", "open"]

    def predict(self, eye_bgr: np.ndarray | None) -> EyePrediction:
        if eye_bgr is None or eye_bgr.size == 0:
            return EyePrediction("UNCERTAIN", 0.0, 0.5, source="NO_ROI")

        prepared = prepare_eye_for_model(
            eye_bgr,
            width=self.image_width,
            height=self.image_height,
        )
        batch = np.expand_dims(prepared, axis=0)
        open_probability = float(self.model.predict(batch, verbose=0).reshape(-1)[0])
        open_probability = float(np.clip(open_probability, 0.0, 1.0))

        if open_probability >= 0.5:
            state = "OPEN"
            confidence = open_probability
        else:
            state = "CLOSED"
            confidence = 1.0 - open_probability

        if confidence < self.confidence_threshold:
            state = "UNCERTAIN"

        return EyePrediction(
            state=state,
            confidence=float(confidence),
            open_probability=open_probability,
            source="CNN",
        )


def combine_eye_predictions(
    left: EyePrediction | None,
    right: EyePrediction | None,
    allow_single_eye_fallback: bool = True,
    single_eye_min_confidence: float = 0.85,
) -> EyePrediction:
    valid = [
        pred
        for pred in (left, right)
        if pred is not None and pred.state in {"OPEN", "CLOSED"}
    ]

    if len(valid) == 2:
        left_pred, right_pred = valid
        mean_open = (left_pred.open_probability + right_pred.open_probability) / 2.0

        if left_pred.state == right_pred.state:
            return EyePrediction(
                state=left_pred.state,
                confidence=float((left_pred.confidence + right_pred.confidence) / 2.0),
                open_probability=float(mean_open),
                source="BOTH_EYES",
            )

        return EyePrediction(
            state="UNCERTAIN",
            confidence=float(max(left_pred.confidence, right_pred.confidence)),
            open_probability=float(mean_open),
            source="DISAGREEMENT",
        )

    if len(valid) == 1 and allow_single_eye_fallback:
        pred = valid[0]
        if pred.confidence >= float(single_eye_min_confidence):
            return EyePrediction(
                state=pred.state,
                confidence=float(pred.confidence * 0.90),
                open_probability=pred.open_probability,
                source="SINGLE_EYE",
            )

    probabilities = [
        pred.open_probability
        for pred in (left, right)
        if pred is not None
    ]
    mean_open = float(np.mean(probabilities)) if probabilities else 0.5
    best_confidence = max(
        (pred.confidence for pred in (left, right) if pred is not None),
        default=0.0,
    )
    return EyePrediction(
        state="UNCERTAIN",
        confidence=float(best_confidence),
        open_probability=mean_open,
        source="INSUFFICIENT_EYES",
    )
