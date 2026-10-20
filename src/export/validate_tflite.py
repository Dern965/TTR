from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.utils.helpers import load_config, project_path
from src.vision.preprocessing import prepare_eye_for_model


def find_test_images(
    processed_path: Path,
    max_images: int,
) -> list[Path]:

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    candidates: list[Path] = []

    test_path = processed_path / "test"

    if test_path.exists():

        for class_name in ("closed", "open"):

            class_path = test_path / class_name

            if not class_path.exists():
                continue

            images = sorted(
                path
                for path in class_path.rglob("*")
                if path.is_file()
                and path.suffix.lower() in extensions
            )

            candidates.extend(images)

    if not candidates:

        all_path = processed_path / "all"

        if all_path.exists():

            for class_name in ("closed", "open"):

                class_path = all_path / class_name

                if not class_path.exists():
                    continue

                images = sorted(
                    path
                    for path in class_path.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in extensions
                )

                candidates.extend(images)

    if not candidates:
        raise FileNotFoundError(
            "No se encontraron imágenes de prueba en "
            "data/processed/test ni data/processed/all."
        )

    if len(candidates) <= max_images:
        return candidates

    indices = np.linspace(
        0,
        len(candidates) - 1,
        max_images,
        dtype=int,
    )

    return [candidates[index] for index in indices]


def prepare_image(
    image_path: Path,
    width: int,
    height: int,
) -> np.ndarray:

    image_bgr = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise ValueError(
            f"No se pudo leer la imagen: {image_path}"
        )

    prepared = prepare_eye_for_model(
        image_bgr,
        width=width,
        height=height,
    )

    return np.expand_dims(
        prepared,
        axis=0,
    ).astype(np.float32)


def predict_keras(
    model: tf.keras.Model,
    batch: np.ndarray,
) -> float:

    output = model.predict(
        batch,
        verbose=0,
    )

    return float(
        np.asarray(output).reshape(-1)[0]
    )


def predict_tflite(
    interpreter: tf.lite.Interpreter,
    input_index: int,
    output_index: int,
    batch: np.ndarray,
) -> float:

    interpreter.set_tensor(
        input_index,
        batch,
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_index
    )

    return float(
        np.asarray(output).reshape(-1)[0]
    )


def label_from_probability(
    open_probability: float,
) -> str:

    return (
        "OPEN"
        if open_probability >= 0.5
        else "CLOSED"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compara predicciones Keras y TFLite."
        )
    )

    parser.add_argument(
        "--keras",
        default="models/cnn_basic.keras",
    )

    parser.add_argument(
        "--tflite",
        default="models/android/cnn_basic.tflite",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
    )

    parser.add_argument(
        "--images",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    width = int(
        config["dataset"]["image_width"]
    )

    height = int(
        config["dataset"]["image_height"]
    )

    processed_path = project_path(
        config["dataset"]["processed_path"]
    )

    keras_path = project_path(args.keras)
    tflite_path = project_path(args.tflite)

    if not keras_path.exists():
        raise FileNotFoundError(
            f"No existe: {keras_path}"
        )

    if not tflite_path.exists():
        raise FileNotFoundError(
            f"No existe: {tflite_path}"
        )

    print("=" * 70)
    print("VALIDACIÓN KERAS VS TFLITE")
    print("=" * 70)

    keras_model = tf.keras.models.load_model(
        keras_path
    )

    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print()
    print("Keras input:", keras_model.input_shape)
    print(
        "TFLite input:",
        input_details[0]["shape"],
        input_details[0]["dtype"],
    )

    print(
        "TFLite output:",
        output_details[0]["shape"],
        output_details[0]["dtype"],
    )

    if input_details[0]["dtype"] != np.float32:
        raise RuntimeError(
            "El TFLite no utiliza entrada Float32."
        )

    expected_shape = np.array(
        [1, height, width, 3],
        dtype=np.int32,
    )

    if not np.array_equal(
        input_details[0]["shape"],
        expected_shape,
    ):
        raise RuntimeError(
            "Forma de entrada inesperada: "
            f"{input_details[0]['shape']} "
            f"(esperado {expected_shape})"
        )

    images = find_test_images(
        processed_path=processed_path,
        max_images=args.images,
    )

    differences: list[float] = []
    label_matches = 0

    print()
    print(
        f"Imágenes seleccionadas: {len(images)}"
    )

    print("-" * 70)

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        batch = prepare_image(
            image_path=image_path,
            width=width,
            height=height,
        )

        keras_probability = predict_keras(
            keras_model,
            batch,
        )

        tflite_probability = predict_tflite(
            interpreter,
            input_details[0]["index"],
            output_details[0]["index"],
            batch,
        )

        difference = abs(
            keras_probability
            - tflite_probability
        )

        keras_label = label_from_probability(
            keras_probability
        )

        tflite_label = label_from_probability(
            tflite_probability
        )

        same_label = (
            keras_label == tflite_label
        )

        if same_label:
            label_matches += 1

        differences.append(difference)

        print(
            f"[{index:02d}] "
            f"{image_path.name}"
        )

        print(
            f"     Keras : "
            f"{keras_probability:.8f} "
            f"({keras_label})"
        )

        print(
            f"     TFLite: "
            f"{tflite_probability:.8f} "
            f"({tflite_label})"
        )

        print(
            f"     Diff  : "
            f"{difference:.10f}"
        )

        print()

    max_difference = max(differences)
    mean_difference = float(
        np.mean(differences)
    )

    print("=" * 70)

    print(
        f"Diferencia máxima : "
        f"{max_difference:.10f}"
    )

    print(
        f"Diferencia promedio: "
        f"{mean_difference:.10f}"
    )

    print(
        f"Etiquetas iguales  : "
        f"{label_matches}/{len(images)}"
    )

    print(
        f"Tolerancia          : "
        f"{args.tolerance}"
    )

    print("=" * 70)

    if (
        max_difference <= args.tolerance
        and label_matches == len(images)
    ):

        print(
            "OK: Keras y TFLite producen "
            "resultados equivalentes."
        )

    else:

        print(
            "ADVERTENCIA: las diferencias "
            "superan la tolerancia o existe "
            "alguna etiqueta distinta."
        )


if __name__ == "__main__":
    main()