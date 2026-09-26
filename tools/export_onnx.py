"""Модели для приложения: .pt → .onnx (без PyTorch в exe) + проверка, что ONNX видит то же, что исходник.

Вход ONNX — 480×640: кадр камеры AD5M 640×480 подаётся без полей, как Ultralytics делает с .pt.

    python tools/export_onnx.py --defects models/defects.pt --scene models/scene.pt --out build/models

Проверка — «та же модель»: объекты, найденные Ultralytics на .pt, должны находиться и ONNX-адаптером
приложения (тот же класс, IoU ≥ 0.5). Уверенности у ONNX и .pt систематически различаются, поэтому
порог тревоги подбирается приёмкой на самом ONNX-файле (benchmark.py --defects build/models/defects.onnx).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.config import ROOT  # noqa: E402
from guard.adapters.onnx_detector import OnnxModel  # noqa: E402


def export(pt: Path, out: Path) -> Path:
    from ultralytics import YOLO
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / pt.name
        shutil.copy2(pt, src)
        onnx = Path(YOLO(str(src)).export(format="onnx", imgsz=(480, 640), dynamic=False, simplify=True, device="cpu", nms=False))
        shutil.copy2(onnx, out)
    return out


def parity(pt: Path, onnx: Path, images: list[Path], conf: float = 0.3) -> tuple[int, int]:
    from ultralytics import YOLO
    ref, mine = YOLO(str(pt)), OnnxModel(onnx)
    total = matched = 0
    for p in images:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        r = ref.predict(img, conf=conf, verbose=False, device="cpu")[0].boxes
        got = mine.predict(img, conf - 0.15)
        for c, s, b in zip(r.cls.tolist(), r.conf.tolist(), r.xyxy.tolist()):
            total += 1
            matched += any(c2 == int(c) and _iou(b, b2) >= 0.5 for c2, s2, b2 in got)
    return matched, total


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defects", type=Path, default=ROOT / "models" / "defects.pt")
    ap.add_argument("--scene", type=Path, default=ROOT / "models" / "scene.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "models")
    ap.add_argument("--min-match", type=float, default=0.9)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    test = ROOT / "data" / "defects" / "test"
    with_defects = [p for p in sorted((test / "images").glob("*.jpg"))
                    if (test / "labels" / f"{p.stem}.txt").exists() and (test / "labels" / f"{p.stem}.txt").read_text().strip()]
    own = sorted((test / "images").glob("own_*.jpg"))
    images = with_defects[::max(1, len(with_defects) // 60)][:60] + own[::20][:20]   # и брак, и кадры AD5M
    report = {}
    for name, pt in (("defects", args.defects), ("scene", args.scene)):
        onnx = export(pt, args.out / f"{name}.onnx")
        m, t = parity(pt, onnx, images)
        share = m / t if t else 1.0
        report[name] = {"source": str(pt), "matched": m, "total": t}
        print(f"{name}: {onnx} — совпало {m} из {t} детекций ({share:.0%})")
        if share < args.min_match:
            raise SystemExit(f"{name}: ONNX расходится с исходной моделью — сборку не продолжаю")
        side = pt.with_suffix(".json")
        if side.exists():
            shutil.copy2(side, args.out / f"{name}.json")
    (args.out / "export.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
