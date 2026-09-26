"""Демо-парк виртуальных принтеров для --demo: настоящие записи AD5M, печати идут по кругу.

AD5M-01 — печать идёт нормально, затем срыв (ком нити); AD5M-02 — чистая печать (стоковая
прошивка); AD5M-03 — чистая печать на Klipper/Moonraker. Отменённая/законченная/поставленная
на паузу печать через минуту начинается заново — демо можно смотреть сколько угодно.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from emulator.servers import PrinterNode
from emulator.virtual import VirtualPrinter, synthetic_frames

log = logging.getLogger("guard.emulator")
RESTART_AFTER_S = 60


def load_frames(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.jpg")) if folder.exists() else []


def timeline(files: list[Path], step_s: float = 10.0):
    return [(i * step_s, (lambda p=p: p.read_bytes())) for i, p in enumerate(files)]


def start_demo_fleet(root: Path, first_host: int = 2) -> list[PrinterNode]:
    clean, spag = load_frames(root / "clean"), load_frames(root / "spaghetti")
    clean2, clean3 = load_frames(root / "clean2") or clean, load_frames(root / "clean3") or clean
    lead = 3 if os.environ.get("GUARD_DEMO_FAST") else 30           # проверка сборки: сбой почти сразу
    plans = [("AD5M-01", "flashforge", (clean[:lead] + spag) if clean and spag else None, 3.0),   # сбой
             ("AD5M-02", "flashforge", clean2 or None, 4.0),                                     # у каждого
             ("AD5M-03 Klipper", "moonraker", clean3 or None, 4.0)]                              # своя печать
    nodes = []
    for k, (name, kind, files, speed) in enumerate(plans):
        frames = timeline(files) if files else synthetic_frames(120, 10.0, bad_from=60 if k == 0 else None)
        p = VirtualPrinter(name, f"SNMQLD9{k + 1:06d}", "demo1234", kind=kind, speed=speed, heat_s=20)
        p.start(frames, f"part_{k + 1}.gcode", total_layers=250)
        try:
            nodes.append(PrinterNode(p, f"127.0.0.{first_host + k}").start())
        except OSError as e:
            log.error("эмулятор %s не запустился: %s", name, e)
    threading.Thread(target=_loop, args=(nodes,), daemon=True, name="demo-loop").start()
    log.info("демо-принтеры: %s", ", ".join(f"{n.printer.name} @ {n.host}" for n in nodes))
    return nodes


def _loop(nodes: list[PrinterNode]) -> None:
    idle_since: dict[str, float] = {}
    while True:
        time.sleep(5)
        for n in nodes:
            p = n.printer
            if p.status in ("completed", "cancel", "paused"):
                t0 = idle_since.setdefault(p.name, time.monotonic())
                if time.monotonic() - t0 > RESTART_AFTER_S:
                    p.start(p.frames, p.file, p.total_layers)
                    idle_since.pop(p.name, None)
            else:
                idle_since.pop(p.name, None)
