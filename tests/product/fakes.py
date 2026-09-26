"""Подделки портов для тестов сервиса без сети."""
from __future__ import annotations

import cv2
import numpy as np

from guard.domain.models import Detection, PrinterInfo, PrinterState, Protocol, Status
from guard.ports import PrinterError


def jpg(value: int) -> bytes:
    return cv2.imencode(".jpg", np.full((48, 64, 3), value, np.uint8))[1].tobytes()


class FakePrinter:
    """Общее состояние «принтера», которое видят драйвер и камера."""

    def __init__(self):
        self.status = Status.PRINTING
        self.bright = False          # «брак» на кадре
        self.offline = False
        self.camera_down = False
        self.commands: list[str] = []


class FakeDriver:
    def __init__(self, p: FakePrinter):
        self.p = p

    def probe(self):
        return PrinterInfo(Protocol.FLASHFORGE, serial="SNFAKE")

    def state(self):
        if self.p.offline:
            raise PrinterError("нет ответа")
        return PrinterState(status=self.p.status, progress=0.5)

    def _act(self, name, to):
        self.p.commands.append(name)
        self.p.status = to

    def pause(self):
        self._act("pause", Status.PAUSED)

    def resume(self):
        self._act("resume", Status.PRINTING)

    def cancel(self):
        self._act("cancel", Status.CANCELLED)

    def set_light(self, on):
        self.p.commands.append(f"light:{on}")

    def close(self):
        pass


class FakeCamera:
    def __init__(self, p: FakePrinter):
        self.p = p

    def snapshot(self):
        if self.p.camera_down:
            raise PrinterError("камера не отвечает")
        return jpg(230 if self.p.bright else 30)

    def close(self):
        pass


class FakeDetector:
    has_scene = False

    def __init__(self, cls="spaghetti"):
        self.cls = cls
        self.calls = 0

    def detect(self, images):
        self.calls += 1
        return [[Detection("defects", self.cls, 0.9, (1, 1, 10, 10))] if im.mean() > 128 else [] for im in images]


class MemConfig:
    def __init__(self, settings=None, printers=None):
        from guard.domain.policy import Settings
        self.settings, self.printers, self.saves = settings or Settings(), list(printers or []), 0

    def load(self):
        return self.settings, list(self.printers)

    def save(self, settings, printers):
        self.settings, self.printers, self.saves = settings, list(printers), self.saves + 1


class MemEvents:
    def __init__(self):
        self.items = []

    def add(self, e, jpg=None):
        e.image = "img.jpg" if jpg else None
        self.items.append(e)
        return e

    def recent(self, limit=100, printer_id=None):
        items = [e for e in self.items if printer_id in (None, e.printer_id)]
        return items[::-1][:limit]

    def image(self, name):
        return None

    def kinds(self):
        return [e.kind for e in self.items]
