from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

from src.training.common import CLASS_NAMES
from src.utils.helpers import (
    configure_logging,
    ensure_project_directories,
    load_config,
    project_path,
    save_json,
)

LOGGER = logging.getLogger(__name__)


def load_test_dataset(config: dict) -> tf.data.Dataset:
    processed_path = project_path(config["dataset"]["processed_path"])
    image_size = (
        int(config["dataset"]["image_height"]),
        int(config["dataset"]["image_width"]),
    )
    batch_size = int(config["training"]["batch_size"])

    dataset = tf.keras.utils.image_dataset_from_directory(
        processed_path / "test",
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = model.predict(dataset, verbose=1).reshape(-1)
    y_pred = (probabilities >= 0.5).astype(int)

    y_true_parts = []
    for _, labels in dataset:
        y_true_parts.append(labels.numpy().reshape(-1).astype(int))
    y_true = np.concatenate(y_true_parts)
    return y_true, y_pred, probabilities


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["CLOSED", "OPEN"],
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, values_format="d", cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusión - {model_name}")
    fig.tight_layout()
    path = project_path(f"results/confusion_matrix/{model_name}.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_training_curves(model_name: str) -> None:
    history_path = project_path(f"results/metrics/{model_name}_history.csv")
    if not history_path.exists():
        LOGGER.warning("No se encontró historial para graficar: %s", history_path)
        return

    history = pd.read_csv(history_path)
    if "accuracy" in history and "val_accuracy" in history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history.index + 1, history["accuracy"], label="train")
        ax.plot(history.index + 1, history["val_accuracy"], label="validation")
        ax.set_xlabel("Época")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy - {model_name}")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(project_path(f"results/figures/{model_name}_accuracy.png"), dpi=160)
        plt.close(fig)

    if "loss" in history and "val_loss" in history:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history.index + 1, history["loss"], label="train")
        ax.plot(history.index + 1, history["val_loss"], label="validation")
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.set_title(f"Loss - {model_name}")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(project_path(f"results/figures/{model_name}_loss.png"), dpi=160)
        plt.close(fig)


def save_examples(
    dataset: tf.data.Dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    max_per_category: int = 8,
) -> None:
    output_dir = project_path(f"results/examples/{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for batch_images, _ in dataset:
        images.append(batch_images.numpy().astype(np.uint8))
    all_images = np.concatenate(images, axis=0)

    categories = {
        "correct": np.where(y_true == y_pred)[0],
        "false_positive_open": np.where((y_true == 0) & (y_pred == 1))[0],
        "false_negative_open": np.where((y_true == 1) & (y_pred == 0))[0],
    }

    for category, indices in categories.items():
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for count, idx in enumerate(indices[:max_per_category]):
            image = all_images[idx]
            confidence = probabilities[idx] if y_pred[idx] == 1 else 1.0 - probabilities[idx]
            path = category_dir / (
                f"{count:02d}_true-{CLASS_NAMES[y_true[idx]]}_"
                f"pred-{CLASS_NAMES[y_pred[idx]]}_conf-{confidence:.3f}.png"
            )
            tf.keras.utils.save_img(path, image)


def evaluate(model_path: Path, config: dict) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    model_name = model_path.stem
    model = tf.keras.models.load_model(model_path)
    dataset = load_test_dataset(config)

    y_true, y_pred, probabilities = collect_predictions(model, dataset)

    metrics = {
        "model": model_name,
        "model_path": str(model_path),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_samples": int(len(y_true)),
        "positive_class": "open",
        "class_names": CLASS_NAMES,
    }

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["CLOSED", "OPEN"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["CLOSED", "OPEN"],
        zero_division=0,
    )

    save_json(metrics, f"results/metrics/{model_name}_metrics.json")
    save_json(report_dict, f"results/metrics/{model_name}_classification_report.json")
    report_path = project_path(f"results/metrics/{model_name}_classification_report.txt")
    report_path.write_text(report_text, encoding="utf-8")

    predictions_df = pd.DataFrame(
        {
            "true_label": [CLASS_NAMES[i] for i in y_true],
            "predicted_label": [CLASS_NAMES[i] for i in y_pred],
            "open_probability": probabilities,
            "confidence": np.where(y_pred == 1, probabilities, 1.0 - probabilities),
        }
    )
    predictions_df.to_csv(
        project_path(f"results/metrics/{model_name}_test_predictions.csv"),
        index=False,
    )

    save_confusion_matrix(y_true, y_pred, model_name)
    save_training_curves(model_name)
    save_examples(dataset, y_true, y_pred, probabilities, model_name)

    LOGGER.info("Métricas de %s: %s", model_name, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa un modelo sobre el conjunto TEST.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", required=True, help="Ruta al archivo .keras")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    ensure_project_directories(config)
    evaluate(project_path(args.model), config)


if __name__ == "__main__":
    main()
