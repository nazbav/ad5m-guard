"""Клиент локального API FlashForge Adventurer 5M (AD5M / 5M Pro).

Протокол сверен с библиотекой сообщества ff-5mp-api-py (MIT,
github.com/GhostTypes/ff-5mp-api-py) и спецификацией Parallel-7/flashforge-api-docs:
HTTP POST на порт 8898, в JSON-теле serialNumber и checkCode.
  /detail  → {"code": 0, "detail": {"status": "printing", "printProgress": 0.42, "cameraStreamUrl": …}}
  /control → {"payload": {"cmd": "jobCtl_cmd", "args": {"jobID": "", "action": "pause|continue|cancel"}}}
Статусы: ready, busy, calibrate_doing, error, heating, printing, pausing, paused
(на части прошивок — pause), cancel, completed, downloading.
Код ответа 0 — успех; 1 «Access code is different» — неверный код; -2 — не включён LAN-режим.
Нужен режим «только локальная сеть» (Настройки → WiFi → Network mode); код доступа — там же.
Камера — mjpg-streamer на порту 8080: /?action=stream и /?action=snapshot.
Запасной канал (без авторизации, одна сессия): TCP 8899, ~M601 S1 вход, ~M25 пауза, ~M26 стоп.

Проверено на эмуляторе (emulate_printer.py); на живом AD5M — ещё нет.
"""
from __future__ import annotations

from dataclasses import dataclass

import requests

PRINTING = "printing"


class PrinterError(RuntimeError):
    pass


@dataclass
class PrinterSpec:
    name: str
    host: str
    serial: str
    check_code: str
    api_port: int = 8898
    camera: str | None = None      # по умолчанию http://<host>:8080/?action=snapshot

    @property
    def camera_url(self) -> str:
        return self.camera or f"http://{self.host}:8080/?action=snapshot"


class AD5MClient:
    def __init__(self, spec: PrinterSpec, timeout: float = 5.0):
        self.spec = spec
        self.timeout = timeout
        self.session = requests.Session()

    def _post(self, path: str, payload: dict | None = None) -> dict:
        body = {"serialNumber": self.spec.serial, "checkCode": self.spec.check_code}
        if payload is not None:
            body["payload"] = payload
        url = f"http://{self.spec.host}:{self.spec.api_port}{path}"
        try:
            r = self.session.post(url, json=body, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            raise PrinterError(f"{self.spec.name}: {path}: {e}") from e
        if data.get("code") != 0:
            raise PrinterError(f"{self.spec.name}: {path}: {data.get('message', data)}")
        return data

    def detail(self) -> dict:
        """Состояние принтера: status (нормализованный), printProgress (0..1), остальные поля /detail."""
        d = dict(self._post("/detail").get("detail", {}))
        status = str(d.get("status", "unknown"))
        d["status"] = "paused" if status == "pause" else status  # часть прошивок пишет «pause»
        return d

    def status(self) -> str:
        return self.detail()["status"]

    def _job(self, action: str) -> None:
        self._post("/control", {"cmd": "jobCtl_cmd", "args": {"jobID": "", "action": action}})

    def pause(self) -> None:
        self._job("pause")

    def resume(self) -> None:
        self._job("continue")

    def cancel(self) -> None:
        self._job("cancel")

    def snapshot(self) -> bytes:
        try:
            r = self.session.get(self.spec.camera_url, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise PrinterError(f"{self.spec.name}: камера: {e}") from e
        return r.content
