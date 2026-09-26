"""Кадры для демо-режима приложения (--demo): настоящие записи AD5M, прорежённые и сжатые.

    python tools/make_demo.py --out build/demo
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.config import ROOT  # noqa: E402

# подпапка: ([(начало, конец)], шаг в секундах). Сбой 01.03 — ком нити на столе; человек то и дело
# лезет в кадр руками, такие кадры приложение не судит — в демо только отрезки без рук.
CLIPS = {"spaghetti": ([("2025-03-01 20:55:44", "2025-03-01 20:56:21"),
                        ("2025-03-01 20:56:30", "2025-03-01 20:56:50"),
                        ("2025-03-01 20:58:33", "2025-03-01 20:58:44")], 3),
         "clean": ([("2025-04-29 09:42", "2025-04-29 11:09")], 45),        # чистые печати — у каждого
         "clean2": ([("2025-06-14 15:30", "2025-06-14 17:00")], 45),       # эмулированного принтера своя
         "clean3": ([("2025-04-24 16:45", "2025-04-24 18:15")], 45)}


def hand_filter():
    """Отбрасывать кадры с рукой (модель сцены из build/models, если уже экспортирована)."""
    path = ROOT / "build" / "models" / "scene.onnx"
    if not path.exists():
        return lambda img: False
    from guard.adapters.onnx_detector import OnnxModel
    scene = OnnxModel(path)
    return lambda img: any(scene.names[c] == "hand" for c, _, _ in scene.predict(img, 0.4))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "demo")
    args = ap.parse_args(argv)
    from extract_frames import scan
    frames = scan([ROOT.parent], exclude=[ROOT])
    has_hand = hand_filter()
    if args.out.exists():
        shutil.rmtree(args.out)
    for sub, (ranges, step) in CLIPS.items():
        spans = [(datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in ranges]
        sel = sorted((c for c in frames.values() if any(a <= c.taken <= b for a, b in spans)), key=lambda c: c.taken)
        (args.out / sub).mkdir(parents=True)
        last, n = None, 0
        for c in sel:
            if last and (c.taken - last).total_seconds() < step:
                continue
            img = cv2.imdecode(np.frombuffer(c.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.shape[:2] != (480, 640) or has_hand(img):
                continue
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])[1].tofile(str(args.out / sub / f"{n:04d}.jpg"))
            last, n = c.taken, n + 1
        size = sum(p.stat().st_size for p in (args.out / sub).glob("*.jpg")) / 2**20
        print(f"{sub}: {n} кадров, {size:.1f} МБ")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
