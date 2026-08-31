from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from src.utils.helpers import project_path

FIELDNAMES = [
    "timestamp",
    "scenario",
    "camera_source",
    "expected_state",
    "predicted_state",
    "confidence",
    "closed_duration",
    "alert",
    "notes",
]


class TestCSVLogger:
    def __init__(self, path: str | Path = "results/tests.csv") -> None:
        self.path = project_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
                writer.writeheader()

    def log(
        self,
        scenario: str,
        camera_source: int | str,
        expected_state: str,
        predicted_state: str,
        confidence: float,
        closed_duration: float,
        alert: bool,
        notes: str = "",
    ) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "camera_source": str(camera_source),
            "expected_state": expected_state,
            "predicted_state": predicted_state,
            "confidence": f"{confidence:.6f}",
            "closed_duration": f"{closed_duration:.6f}",
            "alert": bool(alert),
            "notes": notes,
        }
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writerow(row)
