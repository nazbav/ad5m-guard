"""Обновить данные: скачать свежие версии из Roboflow и пересобрать оба датасета.

    python update_data.py                     # скачать последние версии и пересобрать
    python update_data.py --no-fetch          # только пересобрать из того, что уже в data/raw
    python update_data.py --project ad5m-frames --project <проект разметки>

Из каждого проекта в data/raw берётся только последняя версия: разные версии
одного проекта — это одни и те же картинки, их нельзя складывать вместе.
Своя разметка из data/labeled/* (см. label_tool.py) добавляется всегда.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import build_dataset
import fetch_roboflow
from ddet.config import CONFIG_DIR, ROOT

RAW = ROOT / "data" / "raw"
LABELED = ROOT / "data" / "labeled"
VERSION_RE = re.compile(r"^(?P<project>.+)_v(?P<version>\d+)_[^_]+\.zip$")


def latest_versions(raw: Path) -> list[Path]:
    """По одному архиву на проект — с наибольшим номером версии."""
    best: dict[str, tuple[int, Path]] = {}
    for z in raw.glob("*.zip"):
        m = VERSION_RE.match(z.name)
        if not m:
            continue
        v = int(m.group("version"))
        if m.group("project") not in best or v > best[m.group("project")][0]:
            best[m.group("project")] = (v, z)
    return [p for _, p in sorted(best.values(), key=lambda x: x[1].name)]


def labeled_sets(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*") if (p / "data.yaml").exists())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--project", action="append", help="проекты Roboflow (по умолчанию ROBOFLOW_PROJECT из .env)")
    args = ap.parse_args(argv)

    if not args.no_fetch:
        for project in args.project or [None]:
            fetch_roboflow.main(["--project", project] if project else [])

    sources = latest_versions(RAW) + labeled_sets(LABELED)
    if not sources:
        raise SystemExit("Нет данных: data/raw пуст и своей разметки нет")
    print("Источники:\n" + "\n".join(f"  {s}" for s in sources))
    for config in ("defects.yaml", "scene.yaml"):
        print(f"\n=== {config}")
        build_dataset.main([str(s) for s in sources] + ["--config", str(CONFIG_DIR / config)])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
