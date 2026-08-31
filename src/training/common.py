from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf

from src.utils.helpers import project_path, save_json

CLASS_NAMES = ["closed", "open"]


def load_datasets(config: dict) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    processed_path = project_path(config["dataset"]["processed_path"])
    image_size = (
        int(config["dataset"]["image_height"]),
        int(config["dataset"]["image_width"]),
    )
    batch_size = int(config["training"]["batch_size"])
    seed = int(config["split"]["random_seed"])

    kwargs = dict(
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        processed_path / "train",
        shuffle=True,
        seed=seed,
        **kwargs,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        processed_path / "validation",
        shuffle=False,
        **kwargs,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        processed_path / "test",
        shuffle=False,
        **kwargs,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    validation_ds = validation_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, validation_ds, test_ds


def data_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomBrightness(0.12),
            tf.keras.layers.RandomContrast(0.12),
            tf.keras.layers.RandomRotation(0.04, fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
            tf.keras.layers.RandomZoom(0.08, 0.08, fill_mode="reflect"),
        ],
        name="train_only_augmentation",
    )


def build_callbacks(config: dict, model_path: Path) -> list[tf.keras.callbacks.Callback]:
    training = config["training"]
    model_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(training["early_stopping_patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(training["reduce_lr_factor"]),
            patience=int(training["reduce_lr_patience"]),
            min_lr=float(training["min_learning_rate"]),
            verbose=1,
        ),
    ]


def save_training_artifacts(
    history: tf.keras.callbacks.History,
    model_name: str,
    model_path: Path,
    input_shape: tuple[int, int, int],
    architecture: str,
) -> None:
    history_df = pd.DataFrame(history.history)
    history_csv = project_path(f"results/metrics/{model_name}_history.csv")
    history_df.to_csv(history_csv, index=False)

    history_json = {
        key: [float(v) for v in values]
        for key, values in history.history.items()
    }
    save_json(history_json, f"results/metrics/{model_name}_history.json")

    metadata = {
        "model_name": model_name,
        "model_path": str(model_path.relative_to(project_path("."))),
        "class_names": CLASS_NAMES,
        "positive_class": "open",
        "input_shape": list(input_shape),
        "architecture": architecture,
        "preprocessing_embedded_in_model": True,
    }
    metadata_path = model_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
