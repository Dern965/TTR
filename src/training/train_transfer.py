from __future__ import annotations

import argparse
import logging

import tensorflow as tf

from src.training.common import (
    build_callbacks,
    data_augmentation,
    load_datasets,
    save_training_artifacts,
)
from src.utils.helpers import (
    configure_logging,
    ensure_project_directories,
    load_config,
    project_path,
    set_global_seed,
)

LOGGER = logging.getLogger(__name__)


def build_mobilenet_transfer(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="eye_image")
    x = data_augmentation()(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="open_probability")(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenet_transfer")
    return model


def train(config: dict) -> None:
    seed = int(config["split"]["random_seed"])
    set_global_seed(seed)

    height = int(config["dataset"]["image_height"])
    width = int(config["dataset"]["image_width"])
    input_shape = (height, width, 3)

    train_ds, validation_ds, _ = load_datasets(config)
    model = build_mobilenet_transfer(input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(config["training"]["learning_rate"])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model_path = project_path("models/mobilenet_transfer.keras")
    callbacks = build_callbacks(config, model_path)

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=int(config["training"]["epochs"]),
        callbacks=callbacks,
        verbose=1,
    )

    # ModelCheckpoint ya conserva el mejor modelo observado en validación.
    save_training_artifacts(
        history=history,
        model_name="mobilenet_transfer",
        model_path=model_path,
        input_shape=input_shape,
        architecture="MobileNetV2 transfer learning with frozen ImageNet base",
    )
    LOGGER.info("Modelo guardado en %s", model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena MobileNetV2 por transferencia.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    ensure_project_directories(config)
    train(config)


if __name__ == "__main__":
    main()
