"""Поиск моментов сбоя в записанных печатях «учителем» Obico — для отбора кадров на разметку.

    python tools/mine_failures.py --out data/to_label/batch2

Модель Obico (The Spaghetti Detective, один класс «failure», YOLOv2 416×416) обучена
на десятках тысяч реальных сбоев с веб-камер. Её веса без явной лицензии — поэтому
она только помогает найти, КАКИЕ кадры размечать, и в продукт не идёт.

По каждой печати: кадр раз в --every секунд → p = сумма уверенностей детекций
(как у Obico) → экспоненциальное сглаживание (span 12, как у Obico) → отрезки,
где сглаженное p выше --thr. Кадры этих отрезков (±--pad секунд) попадают в партию
на разметку, уже размеченные — нет. Рядом: mining.csv со всеми p и график по печатям.
Веса: weights/obico.onnx (URL — в README, раздел «Учитель Obico»).
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.config import ROOT  # noqa: E402
from ddet.frames import split_sessions  # noqa: E402

INPUT = 416
DET_THRESH = 0.08   # как в obico ml_api/server.py
NMS_IOU = 0.45
EWM_SPAN = 12


class Obico:
    def __init__(self, path: Path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 0
        self.s = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.name = self.s.get_inputs()[0].name

    def p(self, img: np.ndarray) -> float:
        x = cv2.resize(img, (INPUT, INPUT), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)[None] / 255.0
        boxes, confs = self.s.run(None, {self.name: x})
        c, b = confs[0, :, 0], boxes[0, :, 0, :]
        keep = c > DET_THRESH
        b, c = b[keep], c[keep]
        order = c.argsort()[::-1]
        sel = []
        while order.size:
            i = order[0]
            sel.append(i)
            rest = order[1:]
            xx1 = np.maximum(b[i, 0], b[rest, 0]); yy1 = np.maximum(b[i, 1], b[rest, 1])
            xx2 = np.minimum(b[i, 2], b[rest, 2]); yy2 = np.minimum(b[i, 3], b[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area = lambda k: (b[k, 2] - b[k, 0]) * (b[k, 3] - b[k, 1])  # noqa: E731
            order = rest[inter / (area(i) + area(rest) - inter + 1e-9) <= NMS_IOU]
        return float(c[sel].sum())


def ewm(values: list[float], span: int = EWM_SPAN) -> list[float]:
    alpha = 2 / (span + 1)
    out, s = [], None
    for v in values:
        s = v if s is None else alpha * v + (1 - alpha) * s
        out.append(s)
    return out


def stretches(times: list[float], smooth: list[float], thr: float) -> list[tuple[float, float, float]]:
    """Отрезки (начало, конец, максимум), где сглаженное p выше порога."""
    out, start, peak = [], None, 0.0
    for t, v in zip(times, smooth):
        if v > thr:
            start = t if start is None else start
            peak = max(peak, v)
        elif start is not None:
            out.append((start, t, peak))
            start, peak = None, 0.0
    if start is not None:
        out.append((start, times[-1], peak))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "weights" / "obico.onnx")
    ap.add_argument("--every", type=float, default=10, help="кадр раз в столько секунд")
    ap.add_argument("--thr", type=float, default=0.38, help="порог сглаженного p (у Obico 0.38 — «похоже на сбой»)")
    ap.add_argument("--pad", type=float, default=120, help="захват вокруг отрезка, секунд")
    ap.add_argument("--per-stretch", type=int, default=40)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} уже есть")

    from extract_frames import labeled_timestamps, scan, thumbnail
    frames = scan([ROOT.parent], exclude=[ROOT])
    done = labeled_timestamps(ROOT / "data" / "raw")
    for batch in (ROOT / "data" / "to_label").glob("batch*/images"):
        done |= {p.stem for p in batch.glob("*.jpg")}
    model = Obico(args.model)
    sessions = split_sessions([(c.taken, c) for c in frames.values()], timedelta(minutes=20))

    (args.out / "images").mkdir(parents=True)
    rows, picked_total, found = [], 0, []
    for s in sessions:
        items = [c for _, c in s]
        if thumbnail(items[0].read()) is None:
            continue  # не камера AD5M
        t0 = items[0].taken
        sampled, last = [], None
        for c in items:
            t = (c.taken - t0).total_seconds()
            if last is None or t - last >= args.every:
                img = cv2.imdecode(np.frombuffer(c.read(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                sampled.append((t, c, model.p(img)))
                last = t
        if not sampled:
            continue
        times = [t for t, _, _ in sampled]
        smooth = ewm([p for _, _, p in sampled])
        for (t, c, p), sm in zip(sampled, smooth):
            rows.append([f"{t0:%Y%m%d_%H%M}", c.name, round(t), round(p, 3), round(sm, 3)])
        for a, b, peak in stretches(times, smooth, args.thr):
            window = [(t, c) for t, c, _ in sampled if a - args.pad <= t <= b + args.pad and Path(c.name).stem not in done]
            if len(window) > args.per_stretch:
                idx = np.linspace(0, len(window) - 1, args.per_stretch).round().astype(int)
                window = [window[i] for i in sorted(set(idx))]
            for _, c in window:
                (args.out / "images" / c.name).write_bytes(c.read())
            picked_total += len(window)
            found.append((f"{t0:%Y-%m-%d %H:%M}", f"{(t0 + timedelta(seconds=a)):%H:%M}",
                          f"{(t0 + timedelta(seconds=b)):%H:%M}", round(peak, 2), len(window)))
        print(f"  {t0:%Y-%m-%d %H:%M}: кадров {len(sampled)}, max p {max(p for _, _, p in sampled):.2f}, "
              f"отрезков {sum(1 for f in found if f[0] == f'{t0:%Y-%m-%d %H:%M}')}")

    with open(args.out / "mining.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["печать", "кадр", "t_s", "p", "p_сглаж"])
        w.writerows(rows)
    with open(args.out / "stretches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["печать", "с", "по", "пик", "кадров"])
        w.writerows(found)
    print(f"\nОтрезков с подозрением на сбой: {len(found)}, кадров на разметку: {picked_total} → {args.out}")
    for row in found:
        print("  ", *row)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
