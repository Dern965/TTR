from __future__ import annotations

import argparse
import logging
import shutil

import cv2
import pandas as pd

from src.data.inspect_dataset import build_summary, inspect_dataset
from src.utils.helpers import (
    configure_logging,
    ensure_project_directories,
    load_config,
    project_path,
    save_json,
)

LOGGER = logging.getLogger(__name__)


def prepare_dataset(config: dict, clean: bool = True) -> pd.DataFrame:
    raw_path = project_path(config["dataset"]["raw_path"])
    processed_path = project_path(config["dataset"]["processed_path"])
    image_width = int(config["dataset"]["image_width"])
    image_height = int(config["dataset"]["image_height"])

    if not raw_path.exists():
        raise FileNotFoundError(f"No existe data/raw: {raw_path}")

    inspection = inspect_dataset(raw_path, config["dataset"]["allowed_extensions"])
    inspection_path = project_path("results/metrics/dataset_inspection.csv")
    inspection_path.parent.mkdir(parents=True, exist_ok=True)
    inspection.to_csv(inspection_path, index=False)
    save_json(build_summary(inspection), "results/metrics/dataset_summary.json")

    all_path = processed_path / "all"
    if clean and all_path.exists():
        shutil.rmtree(all_path)

    for class_name in ("closed", "open"):
        (all_path / class_name).mkdir(parents=True, exist_ok=True)

    usable = inspection[
        (inspection["valid"] == True)  # noqa: E712
        & inspection["label"].isin(["open", "closed"])
        & (inspection["duplicate_of"] == "")
    ].copy()

    if usable.empty:
        raise RuntimeError(
            "No se encontraron imágenes utilizables con etiqueta open/closed. "
            "Para MRL se reconoce automáticamente el nombre oficial; para otros datasets "
            "coloca las imágenes dentro de carpetas open/closed."
        )

    metadata_columns = [
        "mrl_gender",
        "mrl_glasses",
        "mrl_reflection",
        "mrl_lighting",
        "mrl_sensor",
    ]

    output_rows: list[dict] = []
    for idx, row in usable.reset_index(drop=True).iterrows():
        source = raw_path / row["relative_path"]
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            LOGGER.warning("Se omitió una imagen que dejó de ser legible: %s", source)
            continue

        interpolation = (
            cv2.INTER_AREA
            if image.shape[1] >= image_width or image.shape[0] >= image_height
            else cv2.INTER_CUBIC
        )
        resized = cv2.resize(image, (image_width, image_height), interpolation=interpolation)

        label = row["label"]
        short_hash = row["sha256"][:12]
        destination = all_path / label / f"{short_hash}_{source.stem}.jpg"
        if not cv2.imwrite(str(destination), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            LOGGER.warning("No se pudo guardar: %s", destination)
            continue

        output = {
            "processed_path": str(destination.relative_to(processed_path)),
            "label": label,
            "subject_id": row["subject_id"],
            "source_relative_path": row["relative_path"],
            "width": image_width,
            "height": image_height,
            "sha256": row["sha256"],
        }
        for column in metadata_columns:
            output[column] = row.get(column, "")
        output_rows.append(output)

        if (idx + 1) % 1000 == 0:
            LOGGER.info("Preparadas %d imágenes...", idx + 1)

    manifest = pd.DataFrame(output_rows)
    manifest_path = processed_path / "processed_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "prepared_images": int(len(manifest)),
        "class_counts": {
            str(k): int(v)
            for k, v in manifest["label"].value_counts().to_dict().items()
        },
        "image_width": image_width,
        "image_height": image_height,
        "normalization": (
            "Las imágenes se guardan en uint8; la normalización está incluida dentro "
            "de cada modelo para mantener el mismo preprocesamiento en entrenamiento e inferencia."
        ),
        "mrl_metadata_preserved": [
            column for column in metadata_columns if column in manifest.columns
        ],
    }
    save_json(summary, "results/metrics/dataset_preparation_summary.json")
    LOGGER.info("Dataset preparado: %s", manifest_path)
    LOGGER.info("Conteos: %s", summary["class_counts"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Limpia y redimensiona el dataset.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="No elimina data/processed/all antes de preparar.",
    )
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    ensure_project_directories(config)
    prepare_dataset(config, clean=not args.keep_existing)


if __name__ == "__main__":
    main()
