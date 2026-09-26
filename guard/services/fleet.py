"""Сервис парка: опрос принтеров, распознавание, решения, действия. Работает только через порты.

Один цикл (step): параллельно спросить у всех принтеров состояние и, если идёт печать, кадр;
кадры всех печатающих — одним пакетом в детектор; по каждому принтеру — правило тревог;
при тревоге — событие, уведомление и (для классов из stop_on) остановка печати.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

from guard.domain.models import Event, PrinterConfig, PrinterInfo, PrinterState, Protocol, Status
from guard.domain.policy import AlertPolicy, Settings
from guard.ports import (AuthError, CameraFactory, Clock, ConfigStore, Detector, DriverFactory, EventStore,
                         Notifier, PrinterError)
from guard.services.render import annotate, from_jpg, to_jpg

log = logging.getLogger("guard.fleet")
MANUAL = ("pause", "resume", "cancel")
IDLE_FRAME_EVERY = 6          # вне печати — кадр для панели раз в столько циклов


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Worker:
    cfg: PrinterConfig
    driver: object | None = None
    camera: object | None = None
    policy: AlertPolicy | None = None
    state: PrinterState = field(default_factory=PrinterState)
    error: str | None = None
    camera_error: str | None = None
    last_check: str | None = None
    frame_jpg: bytes | None = None
    frame_at: str | None = None
    problems: list[str] = field(default_factory=list)
    last_alert: str | None = None
    last_alert_at: str | None = None
    stopped_by_guard: bool = False
    recent: deque = field(default_factory=lambda: deque(maxlen=61))   # исходные кадры для обучения

    def view(self) -> dict:
        return {**self.cfg.public(), "state": self.state.to_dict(), "error": self.error,
                "camera_error": self.camera_error, "last_check": self.last_check, "frame_at": self.frame_at,
                "has_frame": self.frame_jpg is not None, "problems": list(self.problems),
                "last_alert": self.last_alert, "last_alert_at": self.last_alert_at,
                "stopped_by_guard": self.stopped_by_guard}


class FleetService:
    def __init__(self, config: ConfigStore, events: EventStore, detector: Detector | None,
                 driver_factory: DriverFactory, camera_factory: CameraFactory,
                 notifier: Notifier | None = None, clock: Clock = time.monotonic, workers: int = 32):
        self.config, self.events, self.detector = config, events, detector
        self.detector_state = "ready" if detector is not None else "off"      # loading | ready | off
        self.driver_factory, self.camera_factory = driver_factory, camera_factory
        self.notifier, self.clock = notifier, clock
        self.settings, printers = config.load()
        self.workers: dict[str, Worker] = {p.id: Worker(p) for p in printers}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="poll")
        self.cycle = 0
        self.started_at = _now()
        self.last_cycle_ms: float | None = None

    # ------------------------------------------------------------------ конфигурация
    def _save(self) -> None:
        self.config.save(self.settings, [w.cfg for w in self.workers.values()])

    def _reset(self, w: Worker) -> None:
        for attr in ("driver", "camera"):
            obj = getattr(w, attr)
            if obj is not None:
                try:
                    obj.close()
                except Exception:          # закрытие не должно ронять сервис
                    pass
            setattr(w, attr, None)
        w.policy, w.error, w.camera_error = None, None, None

    def add_printer(self, data: dict) -> PrinterConfig:
        cfg = PrinterConfig.from_dict({k: v for k, v in data.items() if k != "id"})
        with self.lock:
            if any(w.cfg.name == cfg.name for w in self.workers.values()):
                raise ValueError(f"Принтер с именем «{cfg.name}» уже есть")
            self.workers[cfg.id] = Worker(cfg)
            self._save()
        self._event(self.workers[cfg.id], "info", f"добавлен ({cfg.host}, {cfg.protocol.value})")
        return cfg

    def update_printer(self, pid: str, data: dict) -> PrinterConfig:
        with self.lock:
            w = self._get(pid)
            merged = {**w.cfg.to_dict(), **data, "id": pid}
            if data.get("check_code") == "••••":        # панель не знает код — оставить прежний
                merged["check_code"] = w.cfg.check_code
            cfg = PrinterConfig.from_dict(merged)
            if any(o.cfg.name == cfg.name and o.cfg.id != pid for o in self.workers.values()):
                raise ValueError(f"Принтер с именем «{cfg.name}» уже есть")
            self._reset(w)
            w.cfg = cfg
            self._save()
        self._event(w, "info", "настройки изменены")
        return cfg

    def remove_printer(self, pid: str) -> None:
        with self.lock:
            w = self._get(pid)
            self._reset(w)
            del self.workers[pid]
            self._save()
        self._event(w, "info", "удалён")

    def update_settings(self, data: dict) -> Settings:
        with self.lock:
            merged = {**self.settings.to_dict(), **data}
            if data.get("telegram_token") == "••••":
                merged["telegram_token"] = self.settings.telegram_token
            self.settings = Settings.from_dict(merged)
            for w in self.workers.values():
                w.policy = None                  # новые пороги — новая история
            self._save()
        return self.settings

    def test_connection(self, data: dict) -> tuple[PrinterInfo, PrinterConfig]:
        """Проверка до сохранения: определяет протокол, серийник, адрес камеры."""
        from guard.adapters.drivers import probe
        data = dict(data)
        pid = data.pop("id", None)
        if data.get("check_code") == "••••" and pid in self.workers:     # панель не знает код — взять сохранённый
            data["check_code"] = self.workers[pid].cfg.check_code
        cfg = PrinterConfig.from_dict(data)
        info = probe(cfg)
        cfg.protocol = info.protocol
        if info.serial and not cfg.serial:
            cfg.serial = info.serial
        return info, cfg

    def _get(self, pid: str) -> Worker:
        if pid not in self.workers:
            raise KeyError(f"нет принтера {pid}")
        return self.workers[pid]

    # ------------------------------------------------------------------ управление
    def control(self, pid: str, action: str) -> None:
        if action not in MANUAL:
            raise ValueError(f"действие: {MANUAL}")
        w = self._get(pid)
        driver = self._driver(w)
        getattr(driver, action)()
        if action == "resume":
            w.stopped_by_guard = False
            w.policy = None
        self._event(w, "manual", f"вручную: {action}")

    def training_snapshot(self, pid: str, note: str = "") -> str:
        """Кнопка 📷 на карточке: исходный кадр сейчас + последние кадры до него — в данные для дообучения
        (пропущенный брак, ложная тревога — самое ценное для обучения)."""
        w = self.workers[pid]
        save = getattr(self.events, "save_sample", None)
        if save is None:
            raise PrinterError("хранилище не поддерживает данные для дообучения")
        if w.camera is None:
            w.camera = self.camera_factory(w.cfg)
        try:
            now_jpg = w.camera.snapshot()
        except Exception as e:                   # noqa: BLE001 — камера недоступна: сказать в панели
            raise PrinterError(f"камера: {e}") from e
        before = list(w.recent)[-10:]
        meta = {"manual": True, "note": note.strip()[:500], "printer": w.cfg.name, "host": w.cfg.host,
                "serial": w.cfg.serial, "protocol": w.cfg.protocol.value, "status": w.state.status.value,
                "file": w.state.file, "layer": w.state.layer, "problems_shown": w.problems,
                "frames": [{"when": t, "detections": [{"model": d.model, "class": d.cls, "conf": round(d.conf, 3),
                                                          "box": [round(x, 1) for x in d.box]} for d in dets]}
                           for t, _, dets in before] + [{"when": _now(), "detections": None}]}
        folder = save(f"{w.cfg.name}_вручную", [jpg for _, jpg, _ in before] + [now_jpg], meta)
        self._event(w, "manual", f"снимок для дообучения{': ' + meta['note'] if meta['note'] else ''}")
        return Path(folder).name

    def set_light(self, pid: str, on: bool) -> None:
        w = self._get(pid)
        self._driver(w).set_light(on)
        self._event(w, "manual", f"свет {'вкл' if on else 'выкл'}")

    def _driver(self, w: Worker):
        if w.cfg.protocol is Protocol.AUTO:
            info, cfg = self.test_connection(w.cfg.to_dict())
            with self.lock:
                w.cfg.protocol, w.cfg.serial = cfg.protocol, cfg.serial or w.cfg.serial
                self._save()
            self._event(w, "info", f"определён протокол {info.protocol.value}")
        if w.driver is None:
            w.driver = self.driver_factory(w.cfg)
        return w.driver

    # ------------------------------------------------------------------ события
    def _event(self, w: Worker, kind: str, detail: str, jpg: bytes | None = None) -> Event:
        e = self.events.add(Event(_now(), w.cfg.id, w.cfg.name, kind, detail), jpg)
        log.info("%s %s %s", w.cfg.name, kind, detail)
        if self.notifier and kind in ("alert", "stopped", "error"):
            try:
                self.notifier.notify(e, jpg)
            except Exception as ex:          # уведомления не должны ронять сервис
                log.warning("уведомление не отправлено: %s", ex)
        return e

    # ------------------------------------------------------------------ цикл
    def _poll(self, w: Worker):
        """→ (состояние | None, кадр | None, ошибка принтера | None, ошибка камеры | None)."""
        try:
            state = self._driver(w).state()
        except PrinterError as e:
            if not isinstance(e, AuthError) and w.driver is not None:   # сеть — переподключимся заново
                try:
                    w.driver.close()
                except Exception:
                    pass
                w.driver = None
            return None, None, str(e), None
        want_frame = (state.status.active and w.cfg.ai_enabled) or self.cycle % IDLE_FRAME_EVERY == 0
        if not want_frame:
            return state, None, None, None
        try:
            if w.camera is None:
                w.camera = self.camera_factory(w.cfg)
            img = from_jpg(w.camera.snapshot())
            if img is None:
                return state, None, None, "камера отдала не картинку"
            return state, img, None, None
        except PrinterError as e:
            return state, None, None, str(e)

    def step(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        started = time.perf_counter()
        with self.lock:
            active = [w for w in self.workers.values() if w.cfg.enabled]
        polled = list(self.pool.map(self._poll, active))
        s = self.settings
        judged = []
        for w, (state, img, err, cam_err) in zip(active, polled):
            w.last_check = _now()
            if err:
                if err != w.error:
                    self._event(w, "error", err)
                w.error = err
                w.state = PrinterState(status=Status.OFFLINE, message=err)
                continue
            if w.error:
                self._event(w, "status", "снова на связи")
                w.error = None
            if cam_err != w.camera_error:
                if cam_err:
                    self._event(w, "error", f"камера: {cam_err}")
                w.camera_error = cam_err
            old = w.state.status
            if state.status != old:
                if state.status.active and old not in (Status.PAUSED, Status.PAUSING):
                    w.policy, w.stopped_by_guard, w.problems = None, False, []     # новая печать
                self._event(w, "status", f"{old.value} → {state.status.value}")
            w.state = state
            if img is None:
                continue
            if state.status.active and w.cfg.ai_enabled and self.detector is not None:
                judged.append((w, img))
            else:
                w.frame_jpg, w.frame_at, w.problems = to_jpg(img), _now(), []

        if judged:
            try:
                if hasattr(self.detector, "focus"):        # вторую модель будить только по важным классам
                    self.detector.focus = set(s.stop_on) | set(s.notify_on)
                all_dets = self.detector.detect([img for _, img in judged])
            except Exception as e:
                log.exception("детектор упал")
                all_dets = [[] for _ in judged]
                for w, _ in judged:
                    self._event(w, "error", f"распознавание не удалось: {e}")
            for (w, img), dets in zip(judged, all_dets):
                self._judge(w, img, dets, now, s)
        self.cycle += 1
        self.last_cycle_ms = (time.perf_counter() - started) * 1000

    def _save_training(self, w: Worker, v, s: Settings) -> None:
        """Кадр тревоги и training_frames до него — исходные, с тем, что нашли модели, — для дообучения."""
        save = getattr(self.events, "save_sample", None)
        if not s.training_frames or save is None or not w.recent:
            return
        frames = list(w.recent)[-(s.training_frames + 1):]
        meta = {"printer": w.cfg.name, "host": w.cfg.host, "serial": w.cfg.serial, "protocol": w.cfg.protocol.value,
                "fired": list(v.fired), "stop_for": list(v.stop_for), "file": w.state.file, "layer": w.state.layer,
                "settings": {k: getattr(s, k) for k in ("defect_conf", "scene_conf", "window", "min_hits", "calibration")},
                "frames": [{"when": t, "detections": [{"model": d.model, "class": d.cls, "conf": round(d.conf, 3),
                                                          "box": [round(x, 1) for x in d.box]} for d in dets]}
                           for t, _, dets in frames]}
        try:
            save(f"{w.cfg.name}_{'+'.join(v.fired)}", [jpg for _, jpg, _ in frames], meta)
        except OSError:
            log.exception("не удалось сохранить кадры для обучения")

    def _judge(self, w: Worker, img, dets, now: float, s: Settings) -> None:
        if w.policy is None:
            w.policy = AlertPolicy(s, has_scene=getattr(self.detector, "has_scene", False))
        v = w.policy.update(dets, now)
        header = f"{w.cfg.name}  " + (" ".join(sorted(v.problems)) if v.problems else "ok")
        if s.training_frames:                                     # исходный кадр, без рамок
            w.recent.append((_now(), to_jpg(img, 92), dets))
        shown = [d for d in dets if d not in v.ignored]           # фон печати (тестовая полоска) не рисуем
        shot = annotate(img, shown, s.defect_conf, s.scene_conf, header,
                        alert=", ".join(v.fired) if v.fired else None, stop=bool(v.stop_for))
        w.frame_jpg, w.frame_at, w.problems = to_jpg(shot), _now(), sorted(v.problems)
        if not v.fired:
            return
        what = ", ".join(v.fired)
        w.last_alert, w.last_alert_at = what, _now()
        jpg = to_jpg(shot, 90)
        self._event(w, "alert", what, jpg)
        self._save_training(w, v, s)
        if not v.stop_for or s.action == "none" or w.stopped_by_guard:
            return
        try:
            getattr(w.driver, s.action)()
            w.stopped_by_guard = True
            self._event(w, "stopped", f"{'пауза' if s.action == 'pause' else 'отмена'}: {', '.join(v.stop_for)}", jpg)
        except PrinterError as e:
            self._event(w, "error", f"не удалось остановить печать: {e}", jpg)

    # ------------------------------------------------------------------ для панели
    def view(self) -> dict:
        with self.lock:
            printers = [w.view() for w in self.workers.values()]
        counts: dict[str, int] = {}
        for p in printers:
            key = p["state"]["status"] if p["enabled"] else "disabled"
            counts[key] = counts.get(key, 0) + 1
        return {"started_at": self.started_at, "cycle": self.cycle, "last_cycle_ms": self.last_cycle_ms,
                "settings": self.settings.to_dict(secrets=False), "counts": counts, "printers": printers,
                "detector": self.detector is not None, "detector_state": self.detector_state}

    def frame(self, pid: str) -> bytes | None:
        w = self.workers.get(pid)
        return w.frame_jpg if w else None

    def run_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                self.step()
            except Exception:
                log.exception("ошибка цикла опроса")
            stop.wait(max(0.2, self.settings.interval_s - (time.monotonic() - t0)))

    def close(self) -> None:
        with self.lock:
            for w in self.workers.values():
                self._reset(w)
        self.pool.shutdown(wait=False, cancel_futures=True)
