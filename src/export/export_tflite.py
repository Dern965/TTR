from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tensorflow as tf

from src.utils.helpers import configure_logging, project_path

LOGGER = logging.getLogger(__name__)


def export_tflite(
    keras_model_path: Path,
    output_path: Path,
) -> None:

    if not keras_model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo Keras: {keras_model_path}"
        )

    LOGGER.info("Cargando modelo Keras: %s", keras_model_path)

    model = tf.keras.models.load_model(keras_model_path)

    LOGGER.info("Input shape: %s", model.input_shape)
    LOGGER.info("Output shape: %s", model.output_shape)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Se conserva Float32 para minimizar diferencias
    # entre el modelo Keras y la versión para Android.
    tflite_model = converter.convert()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(tflite_model)

    LOGGER.info(
        "Modelo TFLite guardado en: %s",
        output_path,
    )

    LOGGER.info(
        "Tamaño: %.2f MB",
        output_path.stat().st_size / (1024 * 1024),
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Exporta la CNN a TensorFlow Lite."
    )

    parser.add_argument(
        "--model",
        default="models/cnn_basic.keras",
    )

    parser.add_argument(
        "--output",
        default="models/android/cnn_basic.tflite",
    )

    args = parser.parse_args()

    configure_logging()

    export_tflite(
        keras_model_path=project_path(args.model),
        output_path=project_path(args.output),
    )


if __name__ == "__main__":
    main()