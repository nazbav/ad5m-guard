"""Скачивает версию датасета из Roboflow в data/raw/ (формат YOLO, zip).

    python fetch_roboflow.py                                  # последняя версия проекта из .env
    python fetch_roboflow.py --project ad5m-frames --version 3

Ключ API, рабочее пространство и проект — из .env (ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE,
ROBOFLOW_PROJECT; см. .env.example) или аргументами.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv

from ddet.config import ROOT

API = "https://api.roboflow.com"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", help="рабочее пространство Roboflow (по умолчанию ROBOFLOW_WORKSPACE из .env)")
    ap.add_argument("--project", help="проект Roboflow (по умолчанию ROBOFLOW_PROJECT из .env)")
    ap.add_argument("--version", type=int, help="номер версии (по умолчанию последняя)")
    ap.add_argument("--format", default="yolov11")
    args = ap.parse_args(argv)

    load_dotenv(ROOT / ".env")
    key = os.environ.get("ROBOFLOW_API_KEY")
    args.workspace = args.workspace or os.environ.get("ROBOFLOW_WORKSPACE")
    args.project = args.project or os.environ.get("ROBOFLOW_PROJECT")
    if not (key and args.workspace and args.project):
        raise SystemExit("Нужны ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE и ROBOFLOW_PROJECT: пропишите их в .env (см. .env.example)")

    base = f"{API}/{args.workspace}/{args.project}"
    version = args.version
    if version is None:
        info = requests.get(base, params={"api_key": key}, timeout=30)
        info.raise_for_status()
        version = max(int(v["id"].rsplit("/", 1)[-1]) for v in info.json()["versions"])

    # Первый запрос запускает подготовку выгрузки на стороне Roboflow — ждём ссылку.
    for _ in range(60):
        r = requests.get(f"{base}/{version}/{args.format}", params={"api_key": key}, timeout=60)
        r.raise_for_status()
        link = (r.json().get("export") or {}).get("link")
        if link:
            break
        time.sleep(5)
    else:
        raise SystemExit("Roboflow так и не подготовил выгрузку")

    out = ROOT / "data" / "raw" / f"{args.project}_v{version}_{args.format}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(link, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        tmp = out.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(out)
    print(f"Скачано: {out} ({out.stat().st_size / 2**20:.1f} МБ)")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
