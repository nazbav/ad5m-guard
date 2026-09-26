"""Запуск AD5M Guard: пути, журналы, одна копия, сервис парка, панель, трей.

    AD5M-Guard.exe                 обычный запуск (иконка в трее, панель в браузере)
    AD5M-Guard.exe --demo          демо: встроенные эмулированные принтеры, без реального железа
    AD5M-Guard.exe --no-tray --host 0.0.0.0 --port 8765     как сервер в сети (задайте токен в app.yaml)
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import threading
import time
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path

import requests
import yaml

from guard import __version__

APP = "AD5M-Guard"
log = logging.getLogger("guard")


def bundle_dir() -> Path:
    """Где лежат модели и статика: внутри exe (PyInstaller) или рядом с исходниками."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))


def resource(name: str) -> Path:
    """Папка ресурса: внутри exe — рядом с кодом; из исходников — build/<name> (после export_onnx / make_demo)."""
    # Не «первая существующая»: в корне исходников есть models/ с обучающими .pt — не то.
    return bundle_dir() / name if getattr(sys, "frozen", False) else bundle_dir() / "build" / name


def exe_dir() -> Path:
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]


def data_dir(override: str | None, demo: bool) -> Path:
    """Портативный режим (файл portable.txt рядом с exe) — данные рядом; иначе %LOCALAPPDATA%\\AD5M-Guard."""
    if override:
        base = Path(override)
    elif (exe_dir() / "portable.txt").exists():
        base = exe_dir() / "data"
    else:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP
    return base / "demo" if demo else base


def setup_logging(folder: Path, console: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = RotatingFileHandler(folder / "guard.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if console and sys.stderr:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def load_app_config(path: Path) -> dict:
    default = {"host": "127.0.0.1", "port": 8765, "token": "", "open_browser": True}
    if not path.exists():
        path.write_text("# Панель: адрес и порт. Для доступа из сети — host: 0.0.0.0 и обязательно token.\n"
                        + yaml.safe_dump(default, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return default
    return default | (yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def already_running(url: str) -> bool:
    try:
        return requests.get(url + "api/state", timeout=1.5).ok
    except requests.RequestException:
        return False


def single_instance(data: Path | None = None):
    """Одна копия программы на пользователя Windows — через системный mutex: он виден сразу, ещё пока
    первая копия грузит модели. data — отдельная папка данных (--data-dir, для проверок): своя копия.
    None — уже запущена."""
    key = hashlib.sha1(str(data.resolve()).lower().encode("utf-8")).hexdigest()[:16] if data else "app"
    if os.name == "nt":
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.restype = ctypes.c_void_p
        handle = k32.CreateMutexW(None, False, f"Local\\{APP}-{key}")
        if handle and ctypes.get_last_error() == 183:          # ERROR_ALREADY_EXISTS
            k32.CloseHandle(ctypes.c_void_p(handle))
            return None
        return handle or True                                  # не создался — не мешаем запуску
    import fcntl
    fh = open((data or Path.home()) / f".{APP}-{key}.lock", "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def show_error(text: str, gui: bool) -> None:
    """Программа без консоли: фатальную ошибку показать окном, а не только в журнал."""
    log.error(text)
    if gui and os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, "AD5M Guard", 0x10)


def exclusive_port() -> None:
    """werkzeug включает SO_REUSEADDR, а на Windows это позволяет второй программе занять тот же порт."""
    if os.name == "nt":
        from werkzeug.serving import BaseWSGIServer
        BaseWSGIServer.allow_reuse_address = False


def model_manifest(models: Path) -> dict:
    """models.yaml рядом с моделями: какие файлы и как объединять вторую модель брака (её откалибровали вместе)."""
    m = {"defects": "defects.onnx", "scene": "scene.onnx", "second": None, "fusion": "and", "gate_conf": 0.3,
         "recommended": {}}
    path = models / "models.yaml"
    if path.exists():
        m |= yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return m


def build_detector(models: Path):
    """ONNX-модели из папки models. Без них сервис работает как панель управления, без распознавания."""
    try:
        m = model_manifest(models)
    except (OSError, yaml.YAMLError):
        log.exception("не читается %s — распознавание выключено", models / "models.yaml")
        return None
    defects = models / m["defects"]
    if not defects.exists():
        log.warning("нет %s — распознавание выключено", defects)
        return None
    try:
        from guard.adapters.onnx_detector import OnnxDetector
        det = OnnxDetector(defects, models / m["scene"], second=models / m["second"] if m["second"] else None,
                           fusion=m["fusion"], gate_conf=float(m["gate_conf"]))
        log.info("модели загружены: %s%s", ", ".join(det.models),
                 f" (объединение: {m['fusion']})" if m["second"] else "")
        return det
    except Exception:
        log.exception("модели не загрузились — распознавание выключено")
        return None


def start_demo(data: Path):
    """Встроенные эмулированные принтеры на 127.0.0.2… и готовый конфиг под них."""
    from emulator.fleet import start_demo_fleet
    nodes = start_demo_fleet(resource("demo"))
    cfg = data / "config.yaml"
    if not cfg.exists():
        from guard.adapters.storage import YamlConfigStore
        from guard.domain.models import PrinterConfig
        from guard.domain.policy import Settings
        printers = [PrinterConfig(name=n.printer.name, host=n.host,
                                  protocol="moonraker" if n.printer.kind == "moonraker" else "flashforge",
                                  serial=n.printer.serial if n.printer.kind == "flashforge" else "",
                                  check_code=n.printer.check_code if n.printer.kind == "flashforge" else "")
                    for n in nodes]
        YamlConfigStore(cfg).save(Settings.from_dict({"interval_s": 3}), printers)
    return nodes


def set_autostart(enable: bool) -> None:
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                        winreg.KEY_SET_VALUE) as k:
        if enable:
            winreg.SetValueEx(k, APP, 0, winreg.REG_SZ, f'"{sys.executable}" --no-browser')
        else:
            try:
                winreg.DeleteValue(k, APP)
            except FileNotFoundError:
                pass


def autostart_enabled() -> bool:
    if os.name != "nt":
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as k:
            winreg.QueryValueEx(k, APP)
            return True
    except OSError:
        return False


def run_tray(url: str, data: Path, stop: threading.Event) -> None:
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        log.info("pystray недоступен — без иконки в трее")
        stop.wait()
        return
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((4, 4, 60, 60), 14, fill=(37, 99, 235))
    d.polygon([(32, 14), (50, 22), (50, 34), (32, 52), (14, 34), (14, 22)], fill=(255, 255, 255))
    menu = pystray.Menu(
        pystray.MenuItem("Открыть панель", lambda: webbrowser.open(url), default=True),
        pystray.MenuItem("Папка данных и журналов", lambda: os.startfile(data) if os.name == "nt" else None),
        pystray.MenuItem("Запускать вместе с Windows", lambda i, it: set_autostart(not autostart_enabled()),
                         checked=lambda it: autostart_enabled(), visible=getattr(sys, "frozen", False)),
        pystray.MenuItem("Выход", lambda i, it: (stop.set(), i.stop())))
    icon = pystray.Icon(APP, img, f"AD5M Guard {__version__}", menu)
    threading.Thread(target=lambda: (stop.wait(), icon.stop()), daemon=True).start()
    icon.run()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=APP, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="встроенные эмулированные принтеры")
    ap.add_argument("--data-dir")
    ap.add_argument("--models", type=Path, help="папка с defects.onnx и scene.onnx")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--no-tray", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--version", action="version", version=f"AD5M Guard {__version__}")
    args = ap.parse_args(argv)

    data = data_dir(args.data_dir, args.demo)
    data.mkdir(parents=True, exist_ok=True)
    setup_logging(data / "logs", console=args.no_tray)
    appcfg = load_app_config(data / "app.yaml")
    host, port = args.host or appcfg["host"], args.port or int(appcfg["port"])
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/"
    gui = not args.no_tray
    # Одна копия: mutex (в т.ч. пока первая ещё запускается) и панель, уже отвечающая на этом порту
    # (старая версия без mutex). --data-dir — отдельная копия для проверок сборки.
    lock = single_instance(data if args.data_dir else None)      # держим до выхода: mutex освобождает система
    if lock is None or already_running(url):
        log.info("уже запущен — открываю панель")
        if not args.no_browser:
            webbrowser.open(url)
        return 0
    if host not in ("127.0.0.1", "localhost") and not appcfg.get("token"):
        log.warning("панель открыта в сеть без токена — задайте token в %s", data / "app.yaml")

    from werkzeug.serving import make_server
    from guard.adapters.camera import HttpCamera
    from guard.adapters.drivers import make_driver
    from guard.adapters.notify import TelegramNotifier
    from guard.adapters.storage import SqliteEventStore, YamlConfigStore
    from guard.services.fleet import FleetService
    from guard.web.app import create_app
    from guard.domain.policy import Settings

    log.info("AD5M Guard %s, данные: %s", __version__, data)
    models_dir = args.models or resource("models")
    try:                                    # порог и правило, откалиброванные для этого набора моделей
        from guard.domain import policy
        policy.RECOMMENDED.update(model_manifest(models_dir).get("recommended") or {})
    except (OSError, yaml.YAMLError):
        log.exception("не читается models.yaml")
    nodes = start_demo(data) if args.demo else []      # демо создаёт config.yaml под свои принтеры
    config = YamlConfigStore(data / "config.yaml")
    try:
        settings, _ = config.load()
    except Exception as e:                  # noqa: BLE001 — битый config.yaml: сказать человеку, а не молча упасть
        show_error(f"Не читается {data / 'config.yaml'}:\n{e}\n\nИсправьте или удалите файл.", gui)
        for n in nodes:
            n.stop()
        return 2
    cal = (model_manifest(models_dir).get("calibration") or "") if models_dir.exists() else ""
    if cal and settings.calibration != cal:       # новые модели — порог и правило под них
        from guard.domain import policy
        settings = Settings.from_dict({**settings.to_dict(), **policy.RECOMMENDED, "calibration": cal})
        config.save(settings, config.load()[1])
        log.info("модели обновились (%s) — применены рекомендуемые порог и правило", cal)
    events = SqliteEventStore(data / "events.db", data / "alerts", settings.keep_alert_images_days)
    # Панель — сразу, модели грузятся в фоне (до минуты на слабом ПК): человек видит, что программа запустилась.
    fleet = FleetService(config, events, None, driver_factory=make_driver, camera_factory=HttpCamera.for_printer)
    fleet.detector_state = "loading"
    fleet.notifier = TelegramNotifier(lambda: fleet.settings)
    exclusive_port()
    try:
        server = make_server(host, port, create_app(fleet, appcfg.get("token") or None), threaded=True)
    except (OSError, SystemExit) as e:          # werkzeug при занятом порте делает sys.exit(1)
        show_error(f"Порт {port} занят другой программой ({e}).\n\nЗадайте другой port в {data / 'app.yaml'}.", gui)
        events.close()
        for n in nodes:
            n.stop()
        return 2

    def load_models():
        # Загрузка библиотеки ONNX Runtime держит GIL (на занятом ПК — до 15 с), и веб-поток в это время
        # не отвечает. Даём панели сначала открыться и показать «Загружаю модели…».
        time.sleep(2.0)
        det = build_detector(models_dir)
        fleet.detector, fleet.detector_state = det, ("ready" if det else "off")

    stop = threading.Event()
    threading.Thread(target=load_models, daemon=True, name="models").start()
    threading.Thread(target=server.serve_forever, daemon=True, name="web").start()
    threading.Thread(target=fleet.run_forever, args=(stop,), daemon=True, name="fleet").start()
    log.info("панель: %s", url)
    if appcfg.get("open_browser", True) and not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        if args.no_tray:
            while not stop.is_set():
                time.sleep(0.5)
        else:
            run_tray(url, data, stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.shutdown()
        fleet.close()
        events.close()
        for n in nodes:
            n.stop()
        log.info("остановлен")
    return 0


if __name__ == "__main__":
    sys.exit(main())
