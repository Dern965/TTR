from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.utils.helpers import project_path

LOGGER = logging.getLogger(__name__)


class DebugFrameSaver:
    def __init__(self, config: dict) -> None:
        debug_cfg = config.get("debug", {})
        self.enabled = bool(debug_cfg.get("enabled", False))
        self.save_uncertain = bool(debug_cfg.get("save_uncertain_frames", True))
        self.save_low_confidence = bool(debug_cfg.get("save_low_confidence_frames", True))
        self.save_closed = bool(debug_cfg.get("save_closed_samples", False))
        self.low_confidence_threshold = float(
            debug_cfg.get("low_confidence_threshold", 0.70)
        )
        self.min_interval_seconds = float(debug_cfg.get("min_interval_seconds", 0.75))
        self.max_saved_events = int(debug_cfg.get("max_saved_events", 150))
        self.output_path = project_path(
            debug_cfg.get("output_path", "results/examples/debug")
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._last_saved = float("-inf")
        self._saved_events = 0

    def choose_reason(self, state: str, confidence: float) -> str | None:
        if not self.enabled:
            return None
        if state == "UNCERTAIN" and self.save_uncertain:
            return "uncertain"
        if confidence < self.low_confidence_threshold and self.save_low_confidence:
            return "low_confidence"
        if state == "CLOSED" and self.save_closed:
            return "closed"
        return None

    def maybe_save(
        self,
        frame: np.ndarray,
        left_roi: np.ndarray | None,
        right_roi: np.ndarray | None,
        reason: str | None,
        state: str,
        confidence: float,
        roi_source: str,
    ) -> None:
        if not self.enabled or reason is None:
            return
        if self._saved_events >= self.max_saved_events:
            return

        now = time.monotonic()
        if now - self._last_saved < self.min_interval_seconds:
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        event_dir = self.output_path / f"{stamp}_{reason}_{state.lower()}"
        event_dir.mkdir(parents=True, exist_ok=True)

        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"DEBUG {reason} | {state} | conf={confidence:.3f} | roi={roi_source}",
            (12, max(28, frame.shape[0] - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(event_dir / "frame.jpg"), annotated)

        if left_roi is not None and left_roi.size > 0:
            cv2.imwrite(str(event_dir / "left_eye.jpg"), left_roi)
        if right_roi is not None and right_roi.size > 0:
            cv2.imwrite(str(event_dir / "right_eye.jpg"), right_roi)

        (event_dir / "info.txt").write_text(
            "\n".join(
                [
                    f"reason={reason}",
                    f"state={state}",
                    f"confidence={confidence:.6f}",
                    f"roi_source={roi_source}",
                ]
            ),
            encoding="utf-8",
        )

        self._last_saved = now
        self._saved_events += 1
        LOGGER.debug("Evento debug guardado: %s", event_dir)
