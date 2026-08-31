from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd

from src.utils.helpers import (
    configure_logging,
    ensure_project_directories,
    load_config,
    project_path,
    save_json,
)

LOGGER = logging.getLogger(__name__)

OPEN_WORDS = {"open", "opened", "abierto", "abiertos", "1"}
CLOSED_WORDS = {"closed", "close", "cerrado", "cerrados", "0"}
MRL_PATTERN = re.compile(
    r"^(?P<subject>s\d+)_(?P<image>\d+)_(?P<gender>\d+)_(?P<glasses>\d+)_"
    r"(?P<state>[01])_(?P<reflection>\d+)_(?P<lighting>\d+)_(?P<sensor>\d+)$",
    re.IGNORECASE,
)


def iter_image_files(raw_path: Path, extensions: set[str]) -> Iterable[Path]:
    for path in raw_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mrl_metadata(path: Path) -> dict[str, str]:
    match = MRL_PATTERN.match(path.stem)
    if not match:
        return {}
    groups = match.groupdict()
    return {
        "subject_id": groups["subject"].lower(),
        "label": "open" if groups["state"] == "1" else "closed",
        "mrl_gender": groups["gender"],
        "mrl_glasses": groups["glasses"],
        "mrl_reflection": groups["reflection"],
        "mrl_lighting": groups["lighting"],
        "mrl_sensor": groups["sensor"],
    }


def infer_mrl_metadata(path: Path) -> tuple[str | None, str | None]:
    metadata = parse_mrl_metadata(path)
    if not metadata:
        return None, None
    return metadata["label"], metadata["subject_id"]


def infer_label_from_path(path: Path) -> str | None:
    metadata = parse_mrl_metadata(path)
    if metadata:
        return metadata["label"]

    tokens = [part.lower() for part in path.parts[:-1]] + [path.stem.lower()]
    for token in reversed(tokens):
        normalized = re.sub(r"[^a-záéíóú0-9]+", " ", token).strip()
        words = set(normalized.split())
        if words & OPEN_WORDS:
            return "open"
        if words & CLOSED_WORDS:
            return "closed"
    return None


def infer_subject_id(path: Path) -> str | None:
    metadata = parse_mrl_metadata(path)
    if metadata:
        return metadata["subject_id"]

    for part in reversed(path.parts[:-1]):
        lower = part.lower()
        match = re.fullmatch(r"(?:subject|sujeto|person|persona)[_-]?(\d+)", lower)
        if match:
            return f"subject_{match.group(1)}"
        if re.fullmatch(r"s\d{3,}", lower):
            return lower
    return None


def inspect_dataset(
    raw_path: Path,
    allowed_extensions: Iterable[str],
) -> pd.DataFrame:
    extensions = {ext.lower() for ext in allowed_extensions}
    rows: list[dict] = []
    seen_hashes: dict[str, str] = {}

    for index, image_path in enumerate(sorted(iter_image_files(raw_path, extensions)), start=1):
        relative_path = image_path.relative_to(raw_path)
        mrl_metadata = parse_mrl_metadata(image_path)
        label = infer_label_from_path(image_path)
        subject_id = infer_subject_id(image_path)
        valid = False
        error = ""
        width = height = channels = 0
        file_hash = ""
        duplicate_of = ""

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                raise ValueError("OpenCV no pudo decodificar la imagen.")
            valid = True
            height, width = image.shape[:2]
            channels = 1 if image.ndim == 2 else image.shape[2]
            file_hash = sha256_file(image_path)
            if file_hash in seen_hashes:
                duplicate_of = seen_hashes[file_hash]
            else:
                seen_hashes[file_hash] = str(relative_path)
        except Exception as exc:
            error = str(exc)

        rows.append(
            {
                "relative_path": str(relative_path),
                "label": label or "",
                "subject_id": subject_id or "",
                "valid": valid,
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "sha256": file_hash,
                "duplicate_of": duplicate_of,
                "mrl_gender": mrl_metadata.get("mrl_gender", ""),
                "mrl_glasses": mrl_metadata.get("mrl_glasses", ""),
                "mrl_reflection": mrl_metadata.get("mrl_reflection", ""),
                "mrl_lighting": mrl_metadata.get("mrl_lighting", ""),
                "mrl_sensor": mrl_metadata.get("mrl_sensor", ""),
                "error": error,
            }
        )

        if index % 1000 == 0:
            LOGGER.info("Inspeccionadas %d imágenes...", index)

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_files": 0,
            "valid_images": 0,
            "corrupt_or_invalid": 0,
            "labeled_images": 0,
            "unlabeled_images": 0,
            "duplicates": 0,
            "class_counts": {},
            "unique_subjects_detected": 0,
            "top_dimensions": [],
            "channels": {},
            "mrl_condition_counts": {},
        }

    valid_df = df[df["valid"] == True]  # noqa: E712
    labeled_df = valid_df[valid_df["label"].isin(["open", "closed"])]
    class_counts = labeled_df["label"].value_counts().to_dict()
    dimension_counts = (
        valid_df.groupby(["width", "height"]).size().sort_values(ascending=False).head(20)
        if not valid_df.empty
        else pd.Series(dtype=int)
    )

    subjects = sorted(set(value for value in valid_df["subject_id"] if value))
    mrl_condition_counts: dict[str, dict[str, int]] = {}
    for column in ("mrl_glasses", "mrl_reflection", "mrl_lighting", "mrl_sensor"):
        if column in valid_df:
            values = valid_df.loc[valid_df[column] != "", column].astype(str)
            if not values.empty:
                mrl_condition_counts[column] = {
                    str(k): int(v) for k, v in values.value_counts().to_dict().items()
                }

    return {
        "total_files": int(len(df)),
        "valid_images": int(len(valid_df)),
        "corrupt_or_invalid": int((~df["valid"]).sum()),
        "labeled_images": int(len(labeled_df)),
        "unlabeled_images": int((valid_df["label"] == "").sum()) if not valid_df.empty else 0,
        "duplicates": int((valid_df["duplicate_of"] != "").sum()) if not valid_df.empty else 0,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "unique_subjects_detected": len(subjects),
        "top_dimensions": [
            {"width": int(width), "height": int(height), "count": int(count)}
            for (width, height), count in dimension_counts.items()
        ],
        "channels": {
            str(k): int(v)
            for k, v in Counter(valid_df["channels"].tolist()).items()
        },
        "mrl_condition_counts": mrl_condition_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspecciona el dataset crudo.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    ensure_project_directories(config)

    raw_path = project_path(config["dataset"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"No existe el directorio del dataset: {raw_path}")

    df = inspect_dataset(raw_path, config["dataset"]["allowed_extensions"])
    output_csv = project_path("results/metrics/dataset_inspection.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    summary = build_summary(df)
    save_json(summary, "results/metrics/dataset_summary.json")

    LOGGER.info("Archivos encontrados: %d", summary["total_files"])
    LOGGER.info("Imágenes válidas: %d", summary["valid_images"])
    LOGGER.info("Imágenes por clase: %s", summary["class_counts"])
    LOGGER.info("Duplicados detectados: %d", summary["duplicates"])
    LOGGER.info("Sujetos detectados: %d", summary["unique_subjects_detected"])
    if summary["mrl_condition_counts"]:
        LOGGER.info("Metadata MRL detectada: %s", summary["mrl_condition_counts"])
    LOGGER.info("CSV: %s", output_csv)


if __name__ == "__main__":
    main()
