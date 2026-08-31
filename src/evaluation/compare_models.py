from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.utils.helpers import (
    configure_logging,
    load_config,
    project_path,
    save_config,
)

LOGGER = logging.getLogger(__name__)


def read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def compare(update_config: bool, config_path: str) -> pd.DataFrame:
    candidates = [
        ("cnn_basic", project_path("results/metrics/cnn_basic_metrics.json")),
        ("mobilenet_transfer", project_path("results/metrics/mobilenet_transfer_metrics.json")),
    ]

    rows = []
    for name, metrics_path in candidates:
        metrics = read_metrics(metrics_path)
        if metrics:
            rows.append(
                {
                    "model": name,
                    "accuracy": float(metrics["accuracy"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "model_path": f"models/{name}.keras",
                }
            )

    if not rows:
        raise FileNotFoundError(
            "No hay métricas de modelos. Ejecuta evaluate_model.py para al menos un modelo."
        )

    comparison = (
        pd.DataFrame(rows)
        .sort_values(["f1", "accuracy", "recall"], ascending=False)
        .reset_index(drop=True)
    )
    output = project_path("results/metrics/model_comparison.csv")
    comparison.to_csv(output, index=False)

    best = comparison.iloc[0]
    LOGGER.info(
        "Modelo recomendado por F1 y accuracy: %s (F1=%.4f, accuracy=%.4f)",
        best["model"],
        best["f1"],
        best["accuracy"],
    )

    if update_config:
        config = load_config(config_path)
        config["inference"]["model_path"] = best["model_path"]
        save_config(config, config_path)
        LOGGER.info("config.yaml actualizado: inference.model_path=%s", best["model_path"])

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara las dos alternativas entrenadas.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--update-config", action="store_true")
    args = parser.parse_args()

    configure_logging()
    comparison = compare(args.update_config, args.config)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
