"""Скорость моделей: сколько кадров в секунду и сколько принтеров потянет машина.

    python tools/speed.py --device cpu
    python tools/speed.py --device 0 --batch 16

Берёт настоящие кадры AD5M, прогоняет модели брака и сцены (как сервис) и считает,
сколько принтеров выдержит машина при опросе раз в --interval секунд.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.config import ROOT  # noqa: E402
from ddet.detect import Detector  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defects", type=Path, default=ROOT / "models" / "defects.pt")
    ap.add_argument("--scene", type=Path, default=ROOT / "models" / "scene.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args(argv)

    paths = sorted((ROOT / "data" / "defects" / "test" / "images").glob("own_*.jpg"))[: args.frames]
    images = [cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) for p in paths]
    det = Detector(args.defects, args.scene if args.scene.exists() else None, device=args.device)
    det.batch(images[: args.batch], batch_size=args.batch)          # прогрев
    t0 = time.perf_counter()
    for i in range(0, len(images), args.batch):
        det.batch(images[i:i + args.batch], batch_size=args.batch)
    dt = (time.perf_counter() - t0) / len(images)
    printers = int(args.interval / dt * 0.7)                       # запас 30 % на сеть, JPEG, логику
    print(f"Модели: {', '.join(f'{k}={Path(m.ckpt_path).name}' for k, m in det.models.items())}; "
          f"устройство {args.device}, пакет {args.batch}")
    print(f"{dt * 1000:.0f} мс на кадр (обе модели) → {1 / dt:.1f} кадр/с → "
          f"~{printers} принтеров при опросе раз в {args.interval:g} с")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
