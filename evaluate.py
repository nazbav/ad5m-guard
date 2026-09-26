"""Проверка модели на отложенной выборке.

    python evaluate.py models/defects.pt
    python evaluate.py models/scene.pt --data data/scene/data.yaml --split val

Считает две группы метрик — на всей выборке и отдельно на кадрах AD5M:
  * по рамкам (mAP50, precision, recall) — стандартные метрики Ultralytics;
  * по кадрам — то, от чего зависит остановка печати: на какой доле кадров
    с браком модель его заметила и на какой доле чистых кадров подняла тревогу.
Отчёт кладётся рядом с весами: <веса>.eval.md
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

from ddet.config import ROOT
from ddet.labels import parse_text
from ddet.report import md_to_html, page
from ddet.watch import SCENE_PROBLEMS

THRESHOLDS = (0.25, 0.4, 0.55)


def box_metrics(model, data: Path, split: str, device: str, project: Path, name: str) -> str:
    m = model.val(data=str(data), split=split, imgsz=640, batch=8, device=device, plots=True,
                  project=str(project), name=name, exist_ok=True, verbose=False)
    names = model.names
    rows = [f"| все классы | {m.box.mp:.3f} | {m.box.mr:.3f} | {m.box.map50:.3f} | {m.box.map:.3f} |"]
    for k, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(k)
        rows.append(f"| {names[int(c)]} | {p:.3f} | {r:.3f} | {ap50:.3f} | {ap:.3f} |")
    return "| класс | P | R | mAP50 | mAP50-95 |\n|---|---:|---:|---:|---:|\n" + "\n".join(rows)


def problem_classes(names: dict[int, str]) -> set[int]:
    """Какие классы означают проблему. У модели брака — все; у модели сцены
    стол, голова, тестовая линия и рука — норма, проблема — только SCENE_PROBLEMS."""
    if "pei_plate" in names.values():
        return {i for i, n in names.items() if n in SCENE_PROBLEMS}
    return set(names)


def frame_metrics(model, images: list[Path], device: str) -> str:
    """Метрики по кадрам для нескольких порогов уверенности."""
    names = model.names
    problems = problem_classes(names)
    truth: list[set[int]] = []
    preds: list[list[tuple[int, float]]] = []
    for i in range(0, len(images), 16):
        chunk = images[i:i + 16]
        for img, res in zip(chunk, model.predict([str(p) for p in chunk], conf=min(THRESHOLDS), imgsz=640,
                                                 device=device, verbose=False)):
            label = img.parent.parent / "labels" / f"{img.stem}.txt"
            truth.append({b.cls for b in parse_text(label.read_text(encoding="utf-8"))})
            preds.append([(int(c), float(s)) for c, s in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist())])

    n_pos = sum(bool(t & problems) for t in truth)
    n_neg = len(truth) - n_pos
    out = [f"Кадров: {len(truth)}, из них с проблемой: {n_pos}, без проблем: {n_neg}.",
           "Проблемные классы: " + ", ".join(names[c] for c in sorted(problems)) + ".", "",
           "| порог | проблема замечена (recall) | ложная тревога на кадре без проблем | точность тревоги |",
           "|---:|---:|---:|---:|"]
    per_class: dict[float, Counter] = {}
    for thr in THRESHOLDS:
        tp = fp = 0
        pc = per_class[thr] = Counter()
        for t, p in zip(truth, preds):
            found = {c for c, s in p if s >= thr}
            if t & problems and found & problems:
                tp += 1
            if not t & problems and found & problems:
                fp += 1
            for c in t | found:
                pc[(c, "tp" if c in t and c in found else "fn" if c in t else "fp")] += 1
        precision = tp / max(1, tp + fp)
        out.append(f"| {thr:.2f} | {tp / max(1, n_pos):.1%} | {fp / max(1, n_neg):.1%} | {precision:.1%} |")

    thr = THRESHOLDS[1]
    out += ["", f"По классам при пороге {thr}: на скольких кадрах с этим классом он найден и сколько лишних срабатываний.",
            "", "| класс | кадров с классом | найден | лишних |", "|---|---:|---:|---:|"]
    pc = per_class[thr]
    for c, n in names.items():
        total = pc[(c, "tp")] + pc[(c, "fn")]
        if total or pc[(c, "fp")]:
            out.append(f"| {n} | {total} | {pc[(c, 'tp')] / max(1, total):.0%} | {pc[(c, 'fp')]} |")
    return "\n".join(out)


def split_images(data: Path, split: str) -> list[Path]:
    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    target = data.parent / cfg[split]
    if target.suffix == ".txt":  # список картинок, пути от папки датасета
        lines = target.read_text(encoding="utf-8").split()
        return sorted({(data.parent / l).resolve() for l in lines})
    return sorted(p for p in target.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("--data", type=Path, help="data.yaml датасета (по умолчанию data/<имя модели>/data.yaml)")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--open", action="store_true", help="открыть отчёт в браузере")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    weights = args.weights.resolve()
    model_name = weights.stem if weights.parent.name == "models" else weights.parent.name
    data = (args.data or ROOT / "data" / model_name / "data.yaml").resolve()
    model = YOLO(str(weights))
    project = ROOT / "runs" / model_name / "eval"

    images = split_images(data, args.split)
    own = [p for p in images if p.name.startswith("own_")]
    report = [f"# Проверка {weights.name} на {args.split}", "", f"Датасет: `{data}`", ""]

    table = box_metrics(model, data, args.split, args.device, project, f"{weights.stem}_{args.split}_all")
    report += ["## По рамкам — все картинки", "", table, ""]

    own_list = data.parent / f"{args.split}_own.txt"
    if own and own_list.exists() and len(own) < len(images):
        own_yaml = data.parent / f"_eval_{args.split}_own.yaml"
        cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
        cfg[args.split] = own_list.name
        own_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        table = box_metrics(model, own_yaml, args.split, args.device, project, f"{weights.stem}_{args.split}_own")
        report += ["## По рамкам — только кадры AD5M", "", table, ""]

    report += ["## По кадрам — все картинки", "", frame_metrics(model, images, args.device), ""]
    if own and len(own) < len(images):
        report += ["## По кадрам — только кадры AD5M (главное для продукта)", "", frame_metrics(model, own, args.device), ""]

    text = "\n".join(report)
    out = weights.with_suffix(".eval.md")
    out.write_text(text, encoding="utf-8")
    html_out = weights.with_suffix(".eval.html")
    html_out.write_text(page(f"Проверка {weights.name}", md_to_html(text)), encoding="utf-8")
    print(text)
    print(f"\nОтчёт: {html_out}")
    if args.open and hasattr(os, "startfile"):
        os.startfile(html_out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
