from __future__ import annotations

import logging
import os
import sys
import threading
import time

LOGGER = logging.getLogger(__name__)


class SoundAlerter:
    def __init__(self, enabled: bool, cooldown_seconds: float = 2.0) -> None:
        self.enabled = bool(enabled)
        self.cooldown_seconds = max(0.1, float(cooldown_seconds))
        self._last_alert = float("-inf")
        self._lock = threading.Lock()

    def trigger(self) -> None:
        if not self.enabled:
            return

        now = time.monotonic()
        with self._lock:
            if now - self._last_alert < self.cooldown_seconds:
                return
            self._last_alert = now

        thread = threading.Thread(target=self._play, daemon=True)
        thread.start()

    @staticmethod
    def _play() -> None:
        try:
            if sys.platform.startswith("win"):
                import winsound

                winsound.Beep(1600, 250)
                winsound.Beep(1900, 250)
            else:
                # Campana del terminal: no agrega dependencias y no bloquea el loop.
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception as exc:
            LOGGER.debug("No se pudo emitir sonido: %s", exc)
