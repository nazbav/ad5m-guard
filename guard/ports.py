"""Порты — интерфейсы, через которые сервис парка работает с внешним миром.

Сервис не знает ни протоколов, ни сети, ни файлов: всё это адаптеры. В тестах
вместо адаптеров подставляются подделки, поэтому логика проверяется без принтера.
"""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from guard.domain.models import Detection, Event, PrinterConfig, PrinterInfo, PrinterState
from guard.domain.policy import Settings


class PrinterError(RuntimeError):
    """Принтер недоступен или отказал. Текст — для человека."""


class AuthError(PrinterError):
    """Неверный серийный номер / код доступа."""


class PrinterDriver(Protocol):
    def probe(self) -> PrinterInfo: ...
    def state(self) -> PrinterState: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def cancel(self) -> None: ...
    def set_light(self, on: bool) -> None: ...
    def close(self) -> None: ...


class Camera(Protocol):
    def snapshot(self) -> bytes: ...
    def close(self) -> None: ...


class Detector(Protocol):
    has_scene: bool

    def detect(self, images: list[np.ndarray]) -> list[list[Detection]]: ...


class Notifier(Protocol):
    def notify(self, event: Event, jpg: bytes | None) -> None: ...


class ConfigStore(Protocol):
    def load(self) -> tuple[Settings, list[PrinterConfig]]: ...
    def save(self, settings: Settings, printers: list[PrinterConfig]) -> None: ...


class EventStore(Protocol):
    def add(self, event: Event, jpg: bytes | None) -> Event: ...
    def recent(self, limit: int = 100, printer_id: str | None = None) -> list[Event]: ...
    def image(self, name: str) -> bytes | None: ...


DriverFactory = Callable[[PrinterConfig], PrinterDriver]
CameraFactory = Callable[[PrinterConfig], Camera]
Clock = Callable[[], float]
