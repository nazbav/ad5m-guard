"""RF-DETR (.pth) → ONNX для приложения (без PyTorch), со сверкой с исходной моделью.

    .venv-rfdetr\\Scripts\\python tools\\export_rfdetr.py models\\defects_rfdetr_small_v2real.pth --out build\\models\\defects_rfdetr.onnx

В метаданные ONNX пишутся: arch=rfdetr, names, imgsz, mean, std — по ним guard.adapters.onnx_detector
понимает, как готовить кадр и разбирать выход (dets: cx cy w h 0..1, labels: логиты классов).
Сверка: на кадрах демо объекты .pth и ONNX (через адаптер приложения) должны совпасть ≥ 90%.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}


def match_rate(ref: list, got: list, iou_min: float = 0.5) -> tuple[int, int]:
    from guard.domain.fusion import iou
    hit = 0
    free = list(got)
    for c, _, b in ref:
        m = next((g for g in free if g[0] == c and iou(b, g[2]) >= iou_min), None)
        if m is not None:
            free.remove(m)
            hit += 1
    return hit, len(ref)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "models" / "defects_rfdetr.onnx")
    ap.add_argument("--size", choices=SIZES)
    ap.add_argument("--conf", type=float, default=0.3, help="порог для сверки")
    args = ap.parse_args(argv)

    import onnx
    import rfdetr
    size = args.size or next((s for s in SIZES if f"rfdetr_{s}" in args.weights.stem), "small")
    model = getattr(rfdetr, SIZES[size])(pretrain_weights=str(args.weights), device="cpu")
    names = model.class_names
    names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
    res = int(model.model_config.resolution)
    mean, std = list(model.means), list(model.stds)

    tmp = Path(tempfile.mkdtemp(prefix="rfdetr-onnx-"))
    try:
        path = Path(model.export(output_dir=str(tmp), format="onnx", fp16=False, verbose=False))
        if path.is_dir():
            path = next(path.glob("*.onnx"))
        m = onnx.load(str(path))
        for k, v in {"arch": "rfdetr", "names": repr(names), "imgsz": repr([res, res]),
                     "mean": repr(mean), "std": repr(std), "source": args.weights.name}.items():
            m.metadata_props.add(key=k, value=v)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(m, str(args.out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"ONNX: {args.out} ({args.out.stat().st_size / 2**20:.0f} МБ), вход {res}×{res}, классы {names}")

    # сверка: .pth через rfdetr.predict против ONNX через адаптер приложения
    from ddet.detect import RFDETRBackend
    from guard.adapters.onnx_detector import OnnxModel
    ref_model, onnx_model = RFDETRBackend(args.weights, "cpu"), OnnxModel(args.out)
    frames = sorted((ROOT / "build" / "demo").rglob("*.jpg"))[::3]
    hits = total = 0
    diffs = []
    for p in frames:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        ref = ref_model.predict([img], args.conf)[0]
        got = onnx_model.predict(img, args.conf * 0.7)
        h, t = match_rate(ref, got)
        hits, total = hits + h, total + t
        for c, s, b in ref:
            from guard.domain.fusion import iou
            m = max((g for g in got if g[0] == c), key=lambda g: iou(b, g[2]), default=None)
            if m is not None and iou(b, m[2]) >= 0.5:
                diffs.append(m[1] - s)
    rate = hits / max(total, 1)
    print(f"сверка на {len(frames)} кадрах: совпало объектов {hits}/{total} ({rate:.0%}), "
          f"разница уверенности ONNX−pth: {np.mean(diffs) if diffs else 0:+.3f} ± {np.std(diffs) if diffs else 0:.3f}")
    if total and rate < 0.9:
        raise SystemExit("ONNX расходится с исходной моделью — не использовать")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
