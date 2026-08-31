from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass


class PredictionSmoother:
    def __init__(self, history_size: int = 3, max_uncertain_frames: int = 2) -> None:
        if history_size < 1:
            raise ValueError("history_size debe ser >= 1.")
        if max_uncertain_frames < 0:
            raise ValueError("max_uncertain_frames debe ser >= 0.")
        self.history: deque[str] = deque(maxlen=int(history_size))
        self.max_uncertain_frames = int(max_uncertain_frames)
        self.consecutive_uncertain = 0

    def reset(self) -> None:
        self.history.clear()
        self.consecutive_uncertain = 0

    def update(self, state: str) -> str:
        if state in {"OPEN", "CLOSED"}:
            self.consecutive_uncertain = 0
            self.history.append(state)
        else:
            self.consecutive_uncertain += 1
            if self.consecutive_uncertain > self.max_uncertain_frames:
                self.history.clear()
                return "UNCERTAIN"

        if not self.history:
            return "UNCERTAIN"

        counts = Counter(self.history)
        open_count = counts["OPEN"]
        closed_count = counts["CLOSED"]

        if open_count == closed_count:
            # En empate se conserva el estado válido más reciente para evitar
            # oscilaciones por un solo frame.
            return self.history[-1]
        return "OPEN" if open_count > closed_count else "CLOSED"


@dataclass(frozen=True)
class TemporalStatus:
    state: str
    closed_duration: float
    alert: bool


class TemporalClosureDetector:
    def __init__(
        self,
        threshold_seconds: float = 1.5,
        uncertain_grace_period: float = 0.45,
    ) -> None:
        if threshold_seconds <= 0:
            raise ValueError("threshold_seconds debe ser > 0.")
        if uncertain_grace_period < 0:
            raise ValueError("uncertain_grace_period debe ser >= 0.")

        self.threshold_seconds = float(threshold_seconds)
        self.uncertain_grace_period = float(uncertain_grace_period)
        self.closed_since: float | None = None
        self.uncertain_since: float | None = None

    def reset(self) -> None:
        self.closed_since = None
        self.uncertain_since = None

    def update(self, state: str, now: float | None = None) -> TemporalStatus:
        current_time = time.monotonic() if now is None else float(now)

        if state == "OPEN":
            self.reset()
            return TemporalStatus("OPEN", 0.0, False)

        if state == "CLOSED":
            self.uncertain_since = None
            if self.closed_since is None:
                self.closed_since = current_time
            duration = max(0.0, current_time - self.closed_since)
            return TemporalStatus(
                "CLOSED",
                duration,
                duration >= self.threshold_seconds,
            )

        # Una pérdida breve de detección conserva el inicio del cierre, pero
        # nunca activa una alerta mientras el estado actual sea UNCERTAIN.
        if self.closed_since is None:
            return TemporalStatus("UNCERTAIN", 0.0, False)

        if self.uncertain_since is None:
            self.uncertain_since = current_time

        uncertain_duration = current_time - self.uncertain_since
        if uncertain_duration > self.uncertain_grace_period:
            self.reset()
            return TemporalStatus("UNCERTAIN", 0.0, False)

        duration = max(0.0, current_time - self.closed_since)
        return TemporalStatus("UNCERTAIN", duration, False)
