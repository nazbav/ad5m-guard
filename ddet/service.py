"""Сервис наблюдения за парком принтеров AD5M.

Каждый цикл: параллельно опросить статус и камеру всех принтеров, кадры
печатающих прогнать моделями одним пакетом, по каждому принтеру решить
(PrintWatcher), пора ли тревога, и при тревоге остановить печать.
Состояние парка (последние кадры с рамками, статусы, журнал) доступно
панели управления (ddet/dashboard.py).
"""
from __future__ import annotations

import csv
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import requests

from ddet.detect import Detection, draw
from ddet.printer import PRINTING, AD5MClient, PrinterError, PrinterSpec
from ddet.watch import PrintWatcher, WatchConfig

ACTIONS = ("pause", "cancel", "none")
MANUAL = {"pause": "pause", "resume": "resume", "cancel": "cancel"}


@dataclass
class ServiceConfig:
    interval_s: float = 5.0
    action: str = "pause"             # что делать при тревоге: pause | cancel | none
    only_when_printing: bool = True   # судить только во время печати
    idle_snapshot_every: int = 6      # вне печати кадр для панели — раз в столько циклов (0 — никогда)
    watch: WatchConfig = field(default_factory=WatchConfig)

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f"action должен быть одним из {ACTIONS}")


@dataclass
class Event:
    when: str
    printer: str
    kind: str        # alert | stopped | error | status | manual
    detail: str


class PrinterState:
    def __init__(self, spec: PrinterSpec, cfg: ServiceConfig, has_scene: bool, timeout: float):
        self.spec = spec
        self.client = AD5MClient(spec, timeout=timeout)
        self.cfg = cfg
        self.has_scene = has_scene
        self.watcher = PrintWatcher(cfg.watch, has_scene)
        self.status = "unknown"
        self.error: str | None = None
        # для панели
        self.progress: float | None = None
        self.last_check: str | None = None
        self.frame_jpg: bytes | None = None
        self.frame_at: str | None = None
        self.problems: list[str] = []
        self.last_alert: str | None = None
        self.last_alert_at: str | None = None

    def snapshot(self) -> dict:
        return {"name": self.spec.name, "host": self.spec.host, "status": self.status, "error": self.error,
                "progress": self.progress, "last_check": self.last_check, "frame_at": self.frame_at,
                "has_frame": self.frame_jpg is not None, "problems": list(self.problems),
                "last_alert": self.last_alert, "last_alert_at": self.last_alert_at}


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _jpg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])[1].tobytes()


class Monitor:
    def __init__(self, specs: list[PrinterSpec], detector, cfg: ServiceConfig,
                 notify: Callable[[Event, np.ndarray | None], None] | None = None,
                 workers: int = 32, timeout: float = 5.0):
        has_scene = "scene" in detector.models
        self.printers = [PrinterState(s, cfg, has_scene, timeout) for s in specs]
        self.by_name = {p.spec.name: p for p in self.printers}
        self.detector = detector
        self.cfg = cfg
        self.notify = notify or (lambda e, img: None)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.events: deque[Event] = deque(maxlen=500)
        self.lock = threading.Lock()
        self.cycle = 0
        self.started_at = _now()

    def _poll(self, p: PrinterState):
        """→ (статус, прогресс, кадр, ошибка, судить ли кадр)."""
        try:
            d = p.client.detail()
            status, progress = d["status"], d.get("printProgress")
            printing = status == PRINTING
            idle_shot = (self.cfg.idle_snapshot_every and self.cycle % self.cfg.idle_snapshot_every == 0)
            if self.cfg.only_when_printing and not printing and not idle_shot:
                return status, progress, None, None, False
            img = cv2.imdecode(np.frombuffer(p.client.snapshot(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return status, progress, None, "камера отдала не картинку", False
            return status, progress, img, None, printing or not self.cfg.only_when_printing
        except PrinterError as e:
            return None, None, None, str(e), False

    def _event(self, p: PrinterState, kind: str, detail: str, img=None) -> Event:
        e = Event(_now(), p.spec.name, kind, detail)
        self.events.append(e)
        self.notify(e, img)
        return e

    def step(self, now: float | None = None) -> list[Event]:
        """Один цикл опроса. now — время для истории тревог (в тестах подставляется)."""
        now = time.monotonic() if now is None else now
        events: list[Event] = []
        polled = list(self.pool.map(self._poll, self.printers))
        thresholds = {"defects": self.cfg.watch.defect_conf, "scene": self.cfg.watch.scene_conf}

        judged: list[tuple[PrinterState, np.ndarray]] = []
        with self.lock:
            for p, (status, progress, img, err, judge) in zip(self.printers, polled):
                p.last_check = _now()
                if err:
                    if err != p.error:
                        events.append(self._event(p, "error", err))
                    p.error = err
                    continue
                if p.error:
                    events.append(self._event(p, "status", "снова на связи"))
                    p.error = None
                p.progress = progress
                if status != p.status:
                    if status == PRINTING:
                        p.watcher = PrintWatcher(self.cfg.watch, p.has_scene)  # новая печать — новая история
                        p.problems = []
                    events.append(self._event(p, "status", f"{p.status} → {status}"))
                    p.status = status
                if img is not None and judge:
                    judged.append((p, img))
                elif img is not None:
                    p.frame_jpg, p.frame_at, p.problems = _jpg(img), _now(), []

        if judged:
            all_dets = self.detector.batch([img for _, img in judged])
            for (p, img), dets in zip(judged, all_dets):
                found, fired = p.watcher.update(dets, now)
                header = f"{p.spec.name}  " + (" ".join(sorted(found)) if found else "ok")
                stop_for = [c for c in fired if c not in self.cfg.watch.notify_only]
                shot = draw(img, dets, thresholds, header,
                            alert=", ".join(fired) if fired else None, kind="STOP" if stop_for else "NOTE")
                with self.lock:
                    p.frame_jpg, p.frame_at, p.problems = _jpg(shot), _now(), sorted(found)
                if not fired:
                    continue
                what = ", ".join(fired)
                with self.lock:
                    p.last_alert, p.last_alert_at = what, _now()
                events.append(self._event(p, "alert", what, shot))
                if self.cfg.action == "none" or not stop_for:
                    continue
                try:
                    getattr(p.client, self.cfg.action)()
                    events.append(self._event(p, "stopped", f"{self.cfg.action}: {', '.join(stop_for)}"))
                except PrinterError as e:
                    events.append(self._event(p, "error", f"не удалось {self.cfg.action}: {e}"))
        self.cycle += 1
        return events

    def control(self, name: str, action: str) -> tuple[bool, str]:
        """Ручное управление из панели: pause | resume | cancel."""
        p = self.by_name.get(name)
        if p is None or action not in MANUAL:
            return False, "нет такого принтера или действия"
        try:
            getattr(p.client, MANUAL[action])()
        except PrinterError as e:
            self._event(p, "error", f"вручную {action}: {e}")
            return False, str(e)
        self._event(p, "manual", f"вручную: {action}")
        return True, "ok"

    def state(self) -> dict:
        with self.lock:
            printers = [p.snapshot() for p in self.printers]
        counts: dict[str, int] = {}
        for p in printers:
            key = "error" if p["error"] else p["status"]
            counts[key] = counts.get(key, 0) + 1
        return {"started_at": self.started_at, "cycle": self.cycle, "interval_s": self.cfg.interval_s,
                "action": self.cfg.action, "defect_conf": self.cfg.watch.defect_conf,
                "muted": sorted(self.cfg.watch.muted), "notify_only": sorted(self.cfg.watch.notify_only),
                "counts": counts, "printers": printers,
                "events": [e.__dict__ for e in list(self.events)[-100:]][::-1]}

    def frame(self, name: str) -> bytes | None:
        with self.lock:
            p = self.by_name.get(name)
            return p.frame_jpg if p else None

    def run_forever(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            started = time.monotonic()
            try:
                self.step(started)
            except Exception as e:           # цикл сервиса не должен умирать от одной ошибки
                print(f"Ошибка цикла: {e!r}")
            stop.wait(max(0.0, self.cfg.interval_s - (time.monotonic() - started)))


class EventLog:
    """Уведомления: консоль, events.csv и снимок кадра тревоги; по желанию — Telegram."""

    def __init__(self, out: Path, telegram_token: str | None = None, telegram_chat: str | None = None):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.csv = out / "events.csv"
        if not self.csv.exists():
            self.csv.write_text("when,printer,kind,detail\n", encoding="utf-8")
        self.telegram = (telegram_token, telegram_chat) if telegram_token and telegram_chat else None

    def __call__(self, e: Event, img: np.ndarray | None) -> None:
        print(f"{e.when}  {e.printer:<16} {e.kind:<8} {e.detail}")
        with open(self.csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([e.when, e.printer, e.kind, e.detail])
        jpg = None
        if img is not None:
            jpg = cv2.imencode(".jpg", img)[1].tobytes()
            (self.out / f"{e.when.replace(':', '-').replace(' ', '_')}_{e.printer}.jpg").write_bytes(jpg)
        if self.telegram and e.kind in ("alert", "stopped", "error"):
            token, chat = self.telegram
            text = f"{e.printer}: {e.kind} — {e.detail}"
            try:
                if jpg:
                    requests.post(f"https://api.telegram.org/bot{token}/sendPhoto",
                                  data={"chat_id": chat, "caption": text}, files={"photo": jpg}, timeout=10)
                else:
                    requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                  data={"chat_id": chat, "text": text}, timeout=10)
            except requests.RequestException as ex:
                print(f"Telegram недоступен: {ex}")
