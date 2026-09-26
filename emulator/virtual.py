"""Виртуальный принтер: состояние, печать, температуры, неисправности. Без сети — её дают servers.py."""
from __future__ import annotations

import bisect
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

Frames = list[tuple[float, Callable[[], bytes]]]      # (секунды от начала печати, чтение JPEG)


def synthetic_frames(n: int = 120, step_s: float = 3.0, bad_from: int | None = None) -> Frames:
    """Простые кадры для тестов: серый стол; с кадра bad_from — светлое пятно («брак»)."""
    out = []
    for i in range(n):
        img = np.full((480, 640, 3), 60, np.uint8)
        cv2.rectangle(img, (40, 220), (600, 470), (95, 95, 95), -1)
        if bad_from is not None and i >= bad_from:
            cv2.circle(img, (320, 340), 60, (235, 235, 235), -1)
        cv2.putText(img, f"frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        jpg = cv2.imencode(".jpg", img)[1].tobytes()
        out.append((i * step_s, (lambda b=jpg: b)))
    return out


IDLE_JPG = cv2.imencode(".jpg", np.full((480, 640, 3), 40, np.uint8))[1].tobytes()


@dataclass
class Faults:
    offline: bool = False          # принтер не отвечает ни по одному протоколу
    camera_down: bool = False      # камера не отвечает
    latency_s: float = 0.0         # задержка каждого ответа
    lan_only: bool = True          # выключено — HTTP отвечает code -2
    firmware_error: str = ""       # код ошибки прошивки (HTTP errorCode, Moonraker error)


@dataclass
class VirtualPrinter:
    name: str
    serial: str
    check_code: str
    kind: str = "flashforge"            # flashforge | moonraker
    model: str = "Adventurer 5M Pro"
    firmware: str = "3.1.3"
    mac: str = "88:A9:A7:93:86:F1"
    nozzle_c: float = 220.0                # температуры печати (PLA; PETG — 240/85)
    bed_c: float = 60.0
    speed: float = 1.0                  # во сколько раз быстрее реального времени идёт печать
    heat_s: float = 20.0                # нагрев перед печатью (в секундах записи)
    clock: Callable[[], float] = time.monotonic
    faults: Faults = field(default_factory=Faults)
    commands: list[str] = field(default_factory=list)   # что приходило (для тестов и журнала)

    def __post_init__(self):
        self._lock = threading.RLock()
        self.status = "ready"           # ready heating printing pausing paused completed cancel error
        self.file = ""
        self.frames: Frames = []
        self._times: list[float] = []
        self._pos = 0.0                 # секунды печати
        self._since = self.clock()
        self.light = True
        self.total_layers = 0

    # ----------------------------------------------------------------- печать
    def start(self, frames: Frames, file: str = "part.gcode", total_layers: int = 200) -> None:
        with self._lock:
            self.frames, self._times = frames, [t for t, _ in frames]
            self.file, self.total_layers = file, total_layers
            self._pos, self._since, self.status = -self.heat_s, self.clock(), "heating"

    def _advance(self) -> None:
        if self.status not in ("heating", "printing"):
            return
        now = self.clock()
        self._pos += (now - self._since) * self.speed
        self._since = now
        if self.status == "heating" and self._pos >= 0:
            self.status = "printing"
        if self.frames and self._pos >= self._times[-1]:
            self._pos, self.status = self._times[-1], "completed"

    @property
    def printing_like(self) -> bool:
        return self.status in ("printing", "pausing", "paused")

    def snapshot_state(self) -> dict:
        with self._lock:
            self._advance()
            duration = self._times[-1] if self._times else 1.0
            progress = max(0.0, self._pos) / duration if self.frames else 0.0
            if self.status == "completed":
                progress = 1.0
            nozzle_t = self.nozzle_c if self.status in ("heating", "printing", "pausing", "paused") else 0.0
            bed_t = self.bed_c if nozzle_t else 0.0
            heat_frac = 1.0 if self.status != "heating" else max(0.0, 1 + self._pos / max(self.heat_s, 1e-9))
            return {
                "status": self.status, "file": self.file, "progress": round(progress, 4),
                "layer": int(progress * self.total_layers) if self.frames else 0, "total_layers": self.total_layers,
                "nozzle": round(25 + (nozzle_t - 25) * heat_frac, 1) if nozzle_t else 25.0, "nozzle_target": nozzle_t,
                "bed": round(25 + (bed_t - 25) * heat_frac, 1) if bed_t else 25.0, "bed_target": bed_t,
                "elapsed": max(0.0, self._pos) / self.speed, "remaining": max(0.0, duration - self._pos) / self.speed,
                "light": self.light, "error": self.faults.firmware_error,
            }

    def frame(self) -> bytes:
        with self._lock:
            self._advance()
            if not self.frames:
                return IDLE_JPG
            i = max(0, bisect.bisect_right(self._times, max(0.0, self._pos)) - 1)
            read = self.frames[i][1]
        return read()

    # ----------------------------------------------------------------- управление
    def control(self, action: str) -> bool:
        """pause | resume | cancel. False — действие недопустимо в текущем состоянии."""
        with self._lock:
            self._advance()
            self.commands.append(action)
            if action == "pause" and self.status in ("printing", "heating"):
                self.status = "paused"
            elif action == "resume" and self.status == "paused":
                self._since, self.status = self.clock(), ("printing" if self._pos >= 0 else "heating")
            elif action == "cancel" and self.status in ("printing", "heating", "paused", "pausing"):
                self.status = "cancel"
            else:
                return False
            return True

    def set_light(self, on: bool) -> None:
        with self._lock:
            self.commands.append(f"light:{'on' if on else 'off'}")
            self.light = on

    def delay(self) -> None:
        if self.faults.latency_s:
            time.sleep(self.faults.latency_s)
