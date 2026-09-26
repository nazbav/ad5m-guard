"""Когда слать тревогу: защита от одиночных ложных срабатываний и от спама."""
from __future__ import annotations

from collections import deque


class AlertTracker:
    """Класс поднимает тревогу, если встретился хотя бы в min_hits из последних
    window проверок, и не чаще одного раза в cooldown_s секунд."""

    def __init__(self, window: int, min_hits: int, cooldown_s: float):
        if not 1 <= min_hits <= window:
            raise ValueError("Нужно 1 <= min_hits <= window")
        self.window = window
        self.min_hits = min_hits
        self.cooldown_s = cooldown_s
        self._hits: dict[str, deque[bool]] = {}
        self._last_alert: dict[str, float] = {}

    def update(self, detected: set[str], now: float) -> list[str]:
        for cls in detected:
            self._hits.setdefault(cls, deque(maxlen=self.window))
        fired = []
        for cls, hist in self._hits.items():
            hist.append(cls in detected)
            if sum(hist) < self.min_hits:
                continue
            last = self._last_alert.get(cls)
            if last is not None and now - last < self.cooldown_s:
                continue
            self._last_alert[cls] = now
            fired.append(cls)
        return sorted(fired)

    def hits(self, cls: str) -> int:
        return sum(self._hits.get(cls, ()))
