"""Сравнить модели по выгруженным предсказаниям (tools/dump_predictions.py) одними и теми же метриками.

    python tools/compare.py runs/compare/yolo.json runs/compare/rfdetr.json --out runs/compare/report

Метрики — на всей выборке и отдельно на кадрах AD5M (own_*): AP50 по классам и по кадрам
(замечена ли проблема / ложная тревога) при нескольких порогах.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.metrics import frame_level, map50, per_class_frame_recall  # noqa: E402
from ddet.report import md_to_html, page  # noqa: E402

THRESHOLDS = (0.25, 0.4, 0.55)


def load_truth(data: Path, split: str, files: list[str], names: list[str]) -> dict[str, list]:
    from dump_predictions import split_images
    paths = {p.name: p for p in split_images(data, split)}
    truth = {}
    for f in files:
        p = paths[f]
        w, h = Image.open(p).size
        label = p.parent.parent / "labels" / f"{p.stem}.txt"
        rows = []
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c, xc, yc, bw, bh = line.split()[:5]
                xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
                rows.append([int(c), xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2])
        truth[f] = rows
    return truth


def section(title: str, truth: dict, models: list[dict], names: list[str]) -> list[str]:
    classes = list(range(len(names)))
    lines = [f"## {title}", "", f"Кадров: {len(truth)}", "",
             "| модель | mAP50 | " + " | ".join(names) + " |", "|---|---:|" + "---:|" * len(names)]
    for m in models:
        preds = {f: m["images"].get(f, []) for f in truth}
        ap = map50(truth, preds, classes)
        vals = [v for v in ap.values() if v is not None]
        mean = sum(vals) / len(vals) if vals else 0.0
        lines.append(f"| {m['label']} | {mean:.3f} | " +
                     " | ".join("—" if ap[c] is None else f"{ap[c]:.3f}" for c in classes) + " |")
    lines += ["", "По кадрам: проблема замечена / ложная тревога на чистом кадре.", "",
              "| модель | " + " | ".join(f"порог {t}" for t in THRESHOLDS) + " |", "|---|" + "---|" * len(THRESHOLDS)]
    for m in models:
        preds = {f: m["images"].get(f, []) for f in truth}
        cells = []
        for t in THRESHOLDS:
            r = frame_level(truth, preds, set(classes), t)
            cells.append(f"{r['recall']:.0%} / {r['false_alarm']:.0%}")
        lines.append(f"| {m['label']} | " + " | ".join(cells) + " |")
    lines += ["", f"По классам при пороге {THRESHOLDS[0]}: найдено на кадрах с классом (лишних срабатываний).", "",
              "| модель | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for m in models:
        preds = {f: m["images"].get(f, []) for f in truth}
        s = per_class_frame_recall(truth, preds, THRESHOLDS[0])
        cells = []
        for c in classes:
            n, found, extra = s.get(c, (0, 0, 0))
            cells.append("—" if not n and not extra else f"{found}/{n} ({extra})")
        lines.append(f"| {m['label']} | " + " | ".join(cells) + " |")
    return lines + [""]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="путь без расширения: будут .md и .html")
    args = ap.parse_args(argv)

    models = []
    for p in args.preds:
        m = json.loads(p.read_text(encoding="utf-8"))
        m["label"] = p.stem
        models.append(m)
    names = models[0]["names"]
    for m in models[1:]:
        if m["names"] != names:
            raise SystemExit(f"Разные классы у моделей: {names} и {m['names']}")
    files = sorted(models[0]["images"])
    truth = load_truth(Path(models[0]["data"]), models[0]["split"], files, names)
    own = {f: v for f, v in truth.items() if f.startswith("own_")}
    lines = ["# Сравнение моделей", "", "Модели: " + ", ".join(f"`{m['label']}` ({m['model']})" for m in models), ""]
    if own:
        lines += section("Кадры AD5M (главное)", own, models, names)
    lines += section("Вся выборка", truth, models, names)
    md = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    args.out.with_suffix(".html").write_text(page("Сравнение моделей", md_to_html(md)), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
