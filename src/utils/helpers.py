from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = project_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def save_config(config: dict[str, Any], config_path: str | Path = "config.yaml") -> None:
    path = project_path(config_path)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def ensure_project_directories(config: dict[str, Any] | None = None) -> None:
    paths = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "results" / "metrics",
        PROJECT_ROOT / "results" / "confusion_matrix",
        PROJECT_ROOT / "results" / "figures",
        PROJECT_ROOT / "results" / "examples",
    ]
    if config:
        paths.extend(
            [
                project_path(config["dataset"]["raw_path"]),
                project_path(config["dataset"]["processed_path"]),
            ]
        )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def validate_split_ratios(config: dict[str, Any]) -> tuple[float, float, float]:
    split = config["split"]
    train = float(split["train"])
    validation = float(split["validation"])
    test = float(split["test"])
    total = train + validation + test
    if any(value <= 0 or value >= 1 for value in (train, validation, test)):
        raise ValueError("Los porcentajes train/validation/test deben estar entre 0 y 1.")
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Los porcentajes de división deben sumar 1.0; suma actual={total:.6f}")
    return train, validation, test


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def save_json(data: Any, path: str | Path) -> None:
    output = project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Tipo no serializable: {type(value)!r}")


def parse_camera_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    raise TypeError("camera_source debe ser un entero o una cadena.")


def is_probable_stream(source: int | str) -> bool:
    if isinstance(source, int):
        return True
    lower = source.lower()
    return lower.startswith(("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://"))
