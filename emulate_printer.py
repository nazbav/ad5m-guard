"""Поднимает эмулированные принтеры AD5M, которые проигрывают записанные печати.

    python emulate_printer.py ..\\ff5\\printer_20250418_121536.zip --speed 20
    python emulate_printer.py --all --printers 5 --speed 30

Каждый принтер: камера (как mjpg-streamer AD5M: /?action=stream, /?action=snapshot)
и API управления (как у AD5M на 8898: /detail, /control). Рядом пишется
конфиг для сервиса — запустите в другом окне:

    python monitor.py runs/emulation/printers.yaml

--speed 20 — печать идёт в 20 раз быстрее записи. Ctrl+C — остановить.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import yaml

from ddet.config import ROOT
from ddet.emulator import EmulatedPrinter, Playback, load_session
from ddet.frames import split_sessions


def all_sessions(min_minutes: float) -> list[tuple[str, list]]:
    """Все записанные печати AD5M из родительской папки проекта, длиннее min_minutes."""
    from extract_frames import scan, thumbnail
    found = scan([ROOT.parent], exclude=[ROOT])
    out = []
    for s in split_sessions([(c.taken, c) for c in found.values()], timedelta(minutes=20)):
        start, end = s[0][0], s[-1][0]
        if (end - start).total_seconds() < min_minutes * 60 or thumbnail(s[0][1].read()) is None:
            continue
        out.append((f"{start:%Y%m%d_%H%M}", [((ts - start).total_seconds(), c.read) for ts, c in s]))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help="записи печати: видео, папки или zip с кадрами")
    ap.add_argument("--all", action="store_true", help="взять записанные печати из родительской папки проекта")
    ap.add_argument("--printers", type=int, default=3, help="сколько принтеров поднять с --all")
    ap.add_argument("--min-minutes", type=float, default=30, help="с --all брать печати не короче")
    ap.add_argument("--only", help="с --all: только эти печати через запятую (имена как в сводке run_video)")
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--camera-port", type=int, default=18080, help="порт камеры первого принтера, дальше +1")
    ap.add_argument("--api-port", type=int, default=18898, help="порт API первого принтера, дальше +1")
    ap.add_argument("--config", type=Path, default=ROOT / "runs" / "emulation" / "printers.yaml")
    args = ap.parse_args(argv)

    sessions = [(Path(s).stem, load_session(s)) for s in args.sources]
    if args.all:
        found = all_sessions(args.min_minutes)
        if args.only:
            wanted = {s.strip() for s in args.only.split(",")}
            found = [s for s in found if s[0] in wanted]
        sessions += found[: args.printers]
    if not sessions:
        ap.error("укажите записи или --all")

    emus, specs = [], []
    for i, (name, frames) in enumerate(sessions):
        pname = f"emu-{i + 1:02d}-{name}"
        emu = EmulatedPrinter(Playback(frames, speed=args.speed), serial=f"EMU{i + 1:04d}", check_code="emu",
                              camera_port=args.camera_port + i, api_port=args.api_port + i, name=pname).start()
        emus.append(emu)
        specs.append({"name": pname, "host": "127.0.0.1", "serial": emu.serial, "check_code": emu.check_code,
                      "api_port": emu.api_port, "camera": f"http://127.0.0.1:{emu.camera_port}/?action=snapshot"})
        duration = frames[-1][0] / args.speed / 60
        print(f"{pname}: камера http://127.0.0.1:{emu.camera_port}/?action=stream, "
              f"API :{emu.api_port}, {len(frames)} кадров, ~{duration:.0f} мин при ×{args.speed:g}")

    args.config.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"interval_s": 3, "action": "pause", "printers": specs}
    args.config.write_text("# Сгенерировано emulate_printer.py\n" + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                           encoding="utf-8")
    print(f"\nКонфиг для сервиса: {args.config}\nCtrl+C — остановить эмуляторы")
    try:
        while True:
            time.sleep(30)
            print("  " + "  ".join(f"{e.name.split('-')[1]}:{e.playback.current_status()} {e.playback.progress():.0%}"
                                   for e in emus))
    except KeyboardInterrupt:
        for e in emus:
            e.stop()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
