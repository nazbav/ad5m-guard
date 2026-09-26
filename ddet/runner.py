"""Запуск скрипта проекта в отдельном окне консоли из окна запуска.

    python -m ddet.runner [--open ФАЙЛ] скрипт.py [аргументы...]

Окно не закрывается само: видно, чем кончилось. С --open по успешном
завершении открывается результат (HTML-отчёт, папка).
"""
from __future__ import annotations

import os
import runpy
import sys
import traceback
from pathlib import Path


def run(argv: list[str]) -> bool:
    open_after = None
    if argv[:1] == ["--open"]:
        open_after, argv = argv[1], argv[2:]
    script, args = argv[0], argv[1:]
    if os.name == "nt":
        os.system(f"title {Path(script).stem}")
    sys.argv = [script, *args]
    ok = True
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            ok = False
            if not isinstance(e.code, int):
                print(e.code)
    except KeyboardInterrupt:
        print("\nОстановлено")
    except Exception:
        traceback.print_exc()
        ok = False
    if ok and open_after and Path(open_after).exists() and hasattr(os, "startfile"):
        os.startfile(open_after)
    return ok


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ok = run(sys.argv[1:])
    if not os.environ.get("DDET_NO_PAUSE"):
        input("\nГотово. Enter — закрыть окно…" if ok else "\nОшибка — см. выше. Enter — закрыть окно…")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
