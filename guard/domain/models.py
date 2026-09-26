"""Модели предметной области. Чистые данные, без сети и файлов."""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field, fields
from enum import Enum


class Protocol(str, Enum):
    AUTO = "auto"               # определить при проверке подключения
    FLASHFORGE = "flashforge"   # стоковая прошивка: HTTP 8898 (+ TCP 8899 как запасной канал)
    MOONRAKER = "moonraker"     # Klipper (ZMOD, Klipper Mod): Moonraker 7125


class Status(str, Enum):
    """Нормализованный статус — одинаковый для всех протоколов."""
    IDLE = "idle"
    PRINTING = "printing"
    PAUSING = "pausing"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    HEATING = "heating"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

    @property
    def active(self) -> bool:
        """Идёт печать, которую имеет смысл судить."""
        return self is Status.PRINTING


@dataclass
class PrinterConfig:
    name: str
    host: str
    protocol: Protocol = Protocol.AUTO
    serial: str = ""                 # FlashForge HTTP: серийный номер (TCP ~M115 отдаёт его сам)
    check_code: str = ""             # FlashForge HTTP: код доступа из меню «Только LAN»
    http_port: int = 8898
    tcp_port: int = 8899
    moonraker_port: int = 7125
    camera_url: str = ""             # пусто — по умолчанию для протокола
    enabled: bool = True             # опрашивать принтер
    ai_enabled: bool = True          # распознавать брак
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self):
        self.protocol = Protocol(self.protocol)
        self.name = self.name.strip()
        self.host = self.host.strip()
        if not self.name:
            raise ValueError("Не задано имя принтера")
        if not re.fullmatch(r"[A-Za-z0-9.\-]+", self.host or ""):
            raise ValueError(f"Некорректный адрес принтера: {self.host!r}")
        for p in ("http_port", "tcp_port", "moonraker_port"):
            v = int(getattr(self, p))
            if not 1 <= v <= 65535:
                raise ValueError(f"Некорректный порт {p}: {v}")
            setattr(self, p, v)
        if self.camera_url and not re.match(r"^https?://", self.camera_url):
            raise ValueError("Адрес камеры должен начинаться с http:// или https://")

    def default_camera_url(self) -> str:
        if self.camera_url:
            return self.camera_url
        # И стоковая прошивка AD5M, и моды на Klipper отдают камеру mjpg-streamer на 8080.
        # Для Moonraker точный адрес берётся из /server/webcams/list при «Проверить подключение».
        return f"http://{self.host}:8080/?action=snapshot"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["protocol"] = self.protocol.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PrinterConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def public(self) -> dict:
        """Для панели: без кода доступа."""
        d = self.to_dict()
        d["check_code"] = "••••" if self.check_code else ""
        return d


@dataclass
class PrinterState:
    """Что принтер сообщил о себе при последнем опросе."""
    status: Status = Status.UNKNOWN
    progress: float | None = None        # 0..1
    file: str | None = None
    layer: int | None = None
    total_layers: int | None = None
    nozzle_temp: float | None = None
    nozzle_target: float | None = None
    bed_temp: float | None = None
    bed_target: float | None = None
    elapsed_s: float | None = None
    remaining_s: float | None = None
    light: bool | None = None
    message: str | None = None           # ошибка/сообщение прошивки

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class PrinterInfo:
    """Ответ «проверить подключение»."""
    protocol: Protocol
    model: str = ""
    name: str = ""
    serial: str = ""
    firmware: str = ""
    camera_url: str = ""


@dataclass(frozen=True)
class Detection:
    model: str                                  # "defects" | "scene"
    cls: str
    conf: float
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 в пикселях кадра


@dataclass
class Event:
    when: str
    printer_id: str
    printer: str
    kind: str        # alert | stopped | error | status | manual | info
    detail: str
    image: str | None = None    # имя файла снимка тревоги
