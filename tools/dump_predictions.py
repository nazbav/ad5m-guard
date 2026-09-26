"""Выгрузить предсказания модели на выборку датасета в JSON — для сравнения моделей (tools/compare.py).

    python tools/dump_predictions.py yolo models/defects.pt --out runs/compare/yolo.json
    .venv-rfdetr\\Scripts\\python tools/dump_predictions.py rfdetr models/defects_rfdetr.pth --out runs/compare/rfdetr.json

Работает в обоих окружениях: YOLO — в .venv, RF-DETR — в .venv-rfdetr.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.config import ROOT  # noqa: E402


def split_images(data: Path, split: str) -> list[Path]:
    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    target = data.parent / cfg[split]
    if target.suffix == ".txt":
        return sorted({(data.parent / l).resolve() for l in target.read_text(encoding="utf-8").split()})
    return sorted(p for p in target.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def yolo_predictor(weights: Path, conf: float, device: str):
    from ultralytics import YOLO
    model = YOLO(str(weights))

    def run(paths: list[Path]):
        for p, r in zip(paths, model.predict([str(x) for x in paths], conf=conf, imgsz=640, device=device,
                                             verbose=False, max_det=100)):
            yield p, [[int(c), float(s), *map(float, b)] for c, s, b in
                      zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    return run, [model.names[k] for k in sorted(model.names)]


def rfdetr_predictor(weights: Path, conf: float, size: str):
    import rfdetr
    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[size]
    model = cls(pretrain_weights=str(weights))
    names = list(model.class_names.values()) if isinstance(model.class_names, dict) else list(model.class_names)

    def run(paths: list[Path]):
        for p in paths:
            det = model.predict(Image.open(p).convert("RGB"), threshold=conf)
            yield p, [[int(c), float(s), *map(float, b)] for c, s, b in
                      zip(det.class_id.tolist(), det.confidence.tolist(), det.xyxy.tolist())]
    return run, names


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("framework", choices=("yolo", "rfdetr"))
    ap.add_argument("weights", type=Path)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "defects" / "data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--size", default="small", help="rfdetr: nano | small | medium")
    ap.add_argument("--device", default="0")
    args = ap.parse_args(argv)

    if args.framework == "yolo":
        run, names = yolo_predictor(args.weights, args.conf, args.device)
    else:
        run, names = rfdetr_predictor(args.weights, args.conf, args.size)
    paths = split_images(args.data, args.split)
    images = {}
    for i in range(0, len(paths), 16):
        for p, dets in run(paths[i:i + 16]):
            images[p.name] = dets
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": str(args.weights), "names": names, "data": str(args.data),
                                    "split": args.split, "images": images}), encoding="utf-8")
    print(f"{len(images)} кадров → {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
