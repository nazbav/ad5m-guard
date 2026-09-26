"""Искусственные сбои (см. ddet/synth.py).

    python tools/synth_failures.py cutouts            # библиотека вырезок спагетти → data/synth/cutouts
    python tools/synth_failures.py train --n 2000     # обучающие кадры → data/labeled/synth1

Вырезки: рамки spaghetti из K1 Max (камера в камере принтера) и из своих кадров train/val
(отложенные печати не трогаются), маска — MobileSAM по рамке. Половина исходных картинок
идёт в вырезки для обучения, половина — для приёмки (benchmark.py --synthetic).
Обучающие кадры: нормальные кадры AD5M из train, на пластину вклеено 1–2 вырезки.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.config import ROOT, labeling_names, load_classes, CONFIG_DIR  # noqa: E402
from ddet.labels import parse_text  # noqa: E402
from ddet.synth import cutout_split, distractor, load_library, make_cutout, paste, plate_point  # noqa: E402

CUTS = ROOT / "data" / "synth" / "cutouts"


def spaghetti_boxes(max_per_source: int) -> list[tuple[str, np.ndarray, list]]:
    """(id источника, кадр, [рамки spaghetti в пикселях]) из K1 Max и своих кадров train/val."""
    out = []
    k1 = sorted((ROOT / "data" / "raw").glob("spaghetti-3vuqy_v*_yolov11.zip"))
    if k1:
        z = zipfile.ZipFile(k1[-1])
        names = [n for n in z.namelist() if "/labels/" in n and n.endswith(".txt")]
        taken = 0
        for n in names:
            boxes = parse_text(z.read(n).decode("utf-8"))
            if not boxes:
                continue
            img_name = n.replace("/labels/", "/images/")[:-4] + ".jpg"
            if img_name not in z.namelist():
                continue
            img = cv2.imdecode(np.frombuffer(z.read(img_name), np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            px = [[(b.xc - b.w / 2) * w, (b.yc - b.h / 2) * h, (b.xc + b.w / 2) * w, (b.yc + b.h / 2) * h] for b in boxes]
            out.append((f"k1_{PurePosixPath(n).stem[:24]}", img, px))
            taken += 1
            if taken >= max_per_source:
                break
    ds = ROOT / "data" / "defects"
    names = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))["names"]
    spag = [k for k, v in names.items() if v == "spaghetti"][0]
    taken = 0
    for split in ("train", "val"):
        for lab in sorted((ds / split / "labels").glob("own_*.txt")):
            boxes = [b for b in parse_text(lab.read_text(encoding="utf-8")) if b.cls == spag]
            if not boxes:
                continue
            img = cv2.imdecode(np.fromfile(str(ds / split / "images" / (lab.stem + ".jpg")), np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            px = [[(b.xc - b.w / 2) * w, (b.yc - b.h / 2) * h, (b.xc + b.w / 2) * w, (b.yc + b.h / 2) * h] for b in boxes]
            out.append((f"own_{lab.stem[4:]}", img, px))
            taken += 1
            if taken >= max_per_source:
                break
    return out


def cmd_cutouts(max_per_source: int, device: str) -> None:
    from ultralytics import SAM
    sam = SAM(str(ROOT / "weights" / "mobile_sam.pt"))
    if CUTS.exists():
        shutil.rmtree(CUTS)
    CUTS.mkdir(parents=True)
    stats = {"train": 0, "bench": 0, "rejected": 0}
    for src, img, boxes in spaghetti_boxes(max_per_source):
        split = cutout_split(src)
        for k, box in enumerate(boxes):
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw < 20 or bh < 20:
                stats["rejected"] += 1
                continue
            res = sam(img, bboxes=[box], device=device, verbose=False)[0]
            if res.masks is None or not len(res.masks.data):
                stats["rejected"] += 1
                continue
            mask = res.masks.data[0].cpu().numpy() > 0.5
            fill = mask.sum() / max(bw * bh, 1)
            if not 0.08 <= fill <= 0.9:          # SAM выделил фон или почти ничего — не берём
                stats["rejected"] += 1
                continue
            cut = make_cutout(img, mask.astype(np.uint8))
            if cut is None:
                stats["rejected"] += 1
                continue
            cv2.imencode(".png", cut)[1].tofile(str(CUTS / f"{split}__{src}_{k}.png"))
            stats[split] += 1
    print(f"Вырезок: для обучения {stats['train']}, для приёмки {stats['bench']}, отброшено {stats['rejected']} → {CUTS}")


def cmd_train(n: int, out: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    cuts = load_library(CUTS, "train")
    if not cuts:
        raise SystemExit("Нет вырезок — сначала: synth_failures.py cutouts")
    defects = ROOT / "data" / "defects"
    scene = ROOT / "data" / "scene"
    d_names = yaml.safe_load((defects / "data.yaml").read_text(encoding="utf-8"))["names"]
    s_names = yaml.safe_load((scene / "data.yaml").read_text(encoding="utf-8"))["names"]
    lab_names = labeling_names()
    index = {v: i for i, v in enumerate(lab_names)}
    # Фон — кадры своей камеры из train без спагетти (отложенные печати в train не попадают по построению).
    backgrounds = []
    for lab in sorted((defects / "train" / "labels").glob("own_*.txt")):
        boxes = parse_text(lab.read_text(encoding="utf-8"))
        if any(d_names[b.cls] == "spaghetti" for b in boxes):
            continue
        scene_lab = scene / "train" / "labels" / lab.name
        s_boxes = parse_text(scene_lab.read_text(encoding="utf-8")) if scene_lab.exists() else []
        plates = [b for b in s_boxes if s_names[b.cls] == "pei_plate"]
        if not plates:
            continue
        backgrounds.append((lab.stem, boxes, s_boxes, plates[0]))
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    made = 0
    for k in range(n):
        stem, boxes, s_boxes, plate = backgrounds[rng.integers(len(backgrounds))]
        img = cv2.imdecode(np.fromfile(str(defects / "train" / "images" / (stem + ".jpg")), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        pbox = ((plate.xc - plate.w / 2) * w, (plate.yc - plate.h / 2) * h, (plate.xc + plate.w / 2) * w, (plate.yc + plate.h / 2) * h)
        lines = [f"{index[d_names[b.cls]]} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]
        lines += [f"{index[s_names[b.cls]]} {b.xc:.6f} {b.yc:.6f} {b.w:.6f} {b.h:.6f}" for b in s_boxes]
        # Треть кадров — только «пустышки» (кадр без брака с вклейками), остальное — спагетти (+ иногда пустышка).
        n_spag = 0 if rng.random() < 0.33 else (1 if rng.random() < 0.7 else 2)
        n_dist = 1 if n_spag == 0 or rng.random() < 0.4 else 0
        for kind in ["spaghetti"] * n_spag + ["distractor"] * n_dist:
            cut = cuts[rng.integers(len(cuts))].rgba
            if rng.random() < 0.5:
                cut = cut[:, ::-1]
            if kind == "distractor":
                other = backgrounds[rng.integers(len(backgrounds))][0]
                tex = cv2.imdecode(np.fromfile(str(defects / "train" / "images" / (other + ".jpg")), np.uint8), cv2.IMREAD_COLOR)
                cut = distractor(cut, tex, rng)
            cx, cy = plate_point(pbox, rng)
            width = float(np.clip(rng.lognormal(np.log(110), 0.45), 35, 300))
            img, box = paste(img, cut, cx, cy, width, rng)
            if box and kind == "spaghetti":
                x1, y1, x2, y2 = box
                lines.append(f"{index['spaghetti']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        name = f"synth_printer_{stem[4:]}_{k:05d}"
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(out / "images" / f"{name}.jpg"))
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        made += 1
    (out / "data.yaml").write_text(yaml.safe_dump({"train": "images", "val": "images", "nc": len(lab_names),
                                                   "names": lab_names}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Синтетических кадров: {made} (фонов {len(backgrounds)}, вырезок {len(cuts)}) → {out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("cutouts", "train"))
    ap.add_argument("--max-per-source", type=int, default=700)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "labeled" / "synth1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    if args.command == "cutouts":
        cmd_cutouts(args.max_per_source, args.device)
    else:
        cmd_train(args.n, args.out, args.seed)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
