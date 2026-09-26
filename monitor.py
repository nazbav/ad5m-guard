"""Сервис наблюдения за принтерами AD5M: ловит брак и останавливает печать.

    python monitor.py config/printers.yaml
    python monitor.py runs/emulation/printers.yaml     # против эмуляторов (emulate_printer.py)

Панель управления в браузере: http://127.0.0.1:8765/ (--open — открыть сразу).
Конфиг — см. config/printers.example.yaml. События пишутся в консоль,
runs/monitor/events.csv и снимки тревог туда же. Telegram включается
переменными TELEGRAM_TOKEN и TELEGRAM_CHAT_ID в .env.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

from ddet.config import ROOT
from ddet.detect import Detector
from ddet.modelinfo import recommended_conf
from ddet.printer import PrinterSpec
from ddet.service import EventLog, Monitor, ServiceConfig
from ddet.watch import WatchConfig


def load_config(path: Path) -> tuple[ServiceConfig, list[PrinterSpec]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    watch_keys = WatchConfig.__dataclass_fields__
    kw = {k: v for k, v in data.items() if k in watch_keys}
    for key in ("muted", "notify_only"):
        if key in kw:
            kw[key] = frozenset(kw[key] or ())
    watch = WatchConfig(**kw)
    cfg = ServiceConfig(interval_s=data.get("interval_s", 5), action=data.get("action", "pause"),
                        only_when_printing=data.get("only_when_printing", True), watch=watch)
    specs = [PrinterSpec(**p) for p in data["printers"]]
    if len({s.name for s in specs}) != len(specs):
        raise SystemExit("Имена принтеров в конфиге должны быть разными")
    return cfg, specs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--defects", type=Path, default=ROOT / "models" / "defects.pt")
    ap.add_argument("--scene", type=Path, default=ROOT / "models" / "scene.pt")
    ap.add_argument("--no-scene", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "monitor")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=8765, help="порт панели управления (0 — без панели)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="адрес панели; 0.0.0.0 — доступ из сети (тогда задайте DASHBOARD_TOKEN в .env)")
    ap.add_argument("--open", action="store_true", help="открыть панель в браузере")
    args = ap.parse_args(argv)

    load_dotenv(ROOT / ".env")
    cfg, specs = load_config(args.config)
    if "defect_conf" not in yaml.safe_load(args.config.read_text(encoding="utf-8")):
        cfg.watch.defect_conf = recommended_conf(args.defects, cfg.watch.defect_conf)
    scene = None if args.no_scene or not args.scene.exists() else args.scene
    detector = Detector(args.defects, scene, device=args.device)
    log = EventLog(args.out, os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID"))
    print(f"Принтеров: {len(specs)}, опрос раз в {cfg.interval_s} с, при тревоге: {cfg.action}, "
          f"порог брака {cfg.watch.defect_conf}. "
          f"Модели: {', '.join(detector.models)}. Ctrl+C — стоп.")
    monitor = Monitor(specs, detector, cfg, notify=log)
    if not args.port:
        try:
            monitor.run_forever()
        except KeyboardInterrupt:
            print("Остановлено")
        return

    from werkzeug.serving import make_server
    from ddet.dashboard import create_app
    token = os.environ.get("DASHBOARD_TOKEN")
    if args.host not in ("127.0.0.1", "localhost") and not token:
        print("ВНИМАНИЕ: панель открыта в сеть без DASHBOARD_TOKEN — любой в сети сможет ставить печать на паузу.")
    stop = threading.Event()
    loop = threading.Thread(target=monitor.run_forever, args=(stop,), daemon=True)
    loop.start()
    server = make_server(args.host, args.port, create_app(monitor, token), threaded=True)
    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    print(f"Панель: {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Остановлено")
    finally:
        stop.set()
        server.shutdown()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
