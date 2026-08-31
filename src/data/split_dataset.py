from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.utils.helpers import (
    configure_logging,
    ensure_project_directories,
    load_config,
    project_path,
    save_json,
    validate_split_ratios,
)

LOGGER = logging.getLogger(__name__)


def _stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=(validation_ratio + test_ratio),
        stratify=df["label"],
        random_state=seed,
    )
    relative_test = test_ratio / (validation_ratio + test_ratio)
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return {
        "train": train_df.copy(),
        "validation": validation_df.copy(),
        "test": test_df.copy(),
    }


def _group_split_once(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    groups = df["subject_id"].astype(str)
    first = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(first.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    relative_validation = validation_ratio / (validation_ratio + test_ratio)
    second = GroupShuffleSplit(
        n_splits=1,
        train_size=relative_validation,
        random_state=seed + 1,
    )
    val_idx, test_idx = next(
        second.split(temp_df, groups=temp_df["subject_id"].astype(str))
    )
    return {
        "train": train_df,
        "validation": temp_df.iloc[val_idx].copy(),
        "test": temp_df.iloc[test_idx].copy(),
    }


def _all_splits_have_both_classes(splits: dict[str, pd.DataFrame]) -> bool:
    return all(set(part["label"].unique()) == {"open", "closed"} for part in splits.values())


def _subject_aware_split(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, pd.DataFrame] | None:
    if (df["subject_id"] == "").any():
        return None
    if df["subject_id"].nunique() < 5:
        return None

    for offset in range(100):
        try:
            splits = _group_split_once(
                df,
                train_ratio,
                validation_ratio,
                test_ratio,
                seed + offset,
            )
        except ValueError:
            continue
        if _all_splits_have_both_classes(splits):
            return splits
    return None


def _copy_split_files(
    splits: dict[str, pd.DataFrame],
    processed_path: Path,
) -> pd.DataFrame:
    for split_name in ("train", "validation", "test"):
        split_root = processed_path / split_name
        if split_root.exists():
            shutil.rmtree(split_root)
        for class_name in ("closed", "open"):
            (split_root / class_name).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for split_name, split_df in splits.items():
        for _, row in split_df.iterrows():
            source = processed_path / row["processed_path"]
            destination = (
                processed_path
                / split_name
                / row["label"]
                / Path(row["processed_path"]).name
            )
            shutil.copy2(source, destination)
            rows.append(
                {
                    **row.to_dict(),
                    "split": split_name,
                    "split_path": str(destination.relative_to(processed_path)),
                }
            )
    return pd.DataFrame(rows)


def split_dataset(config: dict) -> pd.DataFrame:
    train_ratio, validation_ratio, test_ratio = validate_split_ratios(config)
    seed = int(config["split"]["random_seed"])
    prefer_subject = bool(config["split"].get("prefer_subject_split", True))
    processed_path = project_path(config["dataset"]["processed_path"])
    manifest_path = processed_path / "processed_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No existe {manifest_path}. Ejecuta primero prepare_dataset.py."
        )

    df = pd.read_csv(manifest_path, dtype={"subject_id": str}).fillna("")
    if df.empty:
        raise RuntimeError("El manifiesto preparado está vacío.")
    if set(df["label"].unique()) != {"open", "closed"}:
        raise RuntimeError("El dataset debe contener ambas clases: open y closed.")

    splits = None
    method = "stratified"
    if prefer_subject:
        splits = _subject_aware_split(
            df,
            train_ratio,
            validation_ratio,
            test_ratio,
            seed,
        )
        if splits is not None:
            method = "subject_group"
            LOGGER.info("Se utilizó división por sujeto para evitar fuga entre particiones.")

    if splits is None:
        splits = _stratified_split(
            df,
            train_ratio,
            validation_ratio,
            test_ratio,
            seed,
        )
        LOGGER.info("Se utilizó división estratificada.")

    split_manifest = _copy_split_files(splits, processed_path)
    split_manifest_path = processed_path / "split_manifest.csv"
    split_manifest.to_csv(split_manifest_path, index=False)

    summary = {
        "method": method,
        "seed": seed,
        "requested_ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "counts": {},
        "subjects_per_split": {},
    }
    for split_name, split_df in splits.items():
        summary["counts"][split_name] = {
            "total": int(len(split_df)),
            "open": int((split_df["label"] == "open").sum()),
            "closed": int((split_df["label"] == "closed").sum()),
        }
        summary["subjects_per_split"][split_name] = int(
            split_df.loc[split_df["subject_id"] != "", "subject_id"].nunique()
        )

    if method == "subject_group":
        subject_sets = {
            name: set(part["subject_id"]) for name, part in splits.items()
        }
        overlaps = {
            "train_validation": sorted(subject_sets["train"] & subject_sets["validation"]),
            "train_test": sorted(subject_sets["train"] & subject_sets["test"]),
            "validation_test": sorted(subject_sets["validation"] & subject_sets["test"]),
        }
        summary["subject_overlaps"] = overlaps

    save_json(summary, "results/metrics/dataset_split_summary.json")
    LOGGER.info("División finalizada: %s", summary["counts"])
    return split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Divide el dataset en train/validation/test.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    configure_logging()
    config = load_config(args.config)
    ensure_project_directories(config)
    split_dataset(config)


if __name__ == "__main__":
    main()
