"""Отбирает кадры AD5M на разметку.

Кадры грабера идут каждые ~3 секунды, и соседние почти одинаковы — размечать
все бессмысленно. Скрипт:
  * собирает кадры printer_ГГГГММДД_ччммсс.jpg из папок и zip-архивов
    (по умолчанию — со всего родительской папки проекта, кроме этого проекта);
  * берёт только кадры камеры AD5M (640×480), повторы одного кадра из разных копий убирает;
  * пропускает кадры, уже размеченные в Roboflow;
  * делит на печати и внутри печати оставляет кадр, только если сцена заметно
    изменилась или прошло достаточно времени;
  * с --prelabel размечает отобранное текущими моделями — остаётся поправить,
    и в первую очередь берёт кадры, где модель что-то подозревает.

Результат — папка в формате YOLO (images/, labels/, data.yaml) и zip рядом:
загружается в Roboflow (Upload → YOLO) или CVAT как есть. Классы — брак и сцена
вместе, см. config/*.yaml.

    python extract_frames.py --out data/to_label/batch1 --prelabel
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import yaml

from ddet.config import ROOT, labeling_names
from ddet.frames import TS_RE, parse_timestamp, split_sessions

CAMERA_SIZE = (640, 480)  # камера AD5M
THUMB = (80, 60)


@dataclass
class Candidate:
    taken: datetime
    name: str
    read: Callable[[], bytes]
    thumb: np.ndarray | None = None
    score: float = 0.0


def scan(sources: list[Path], exclude: list[Path]) -> dict[str, Candidate]:
    """Все кадры грабера из папок и zip. Ключ — время съёмки: копии одного кадра схлопываются."""
    found: dict[str, Candidate] = {}
    excl = [e.resolve() for e in exclude]

    def add(name: str, read):
        ts = parse_timestamp(name)
        if ts is None:
            return
        key = TS_RE.search(name).group(0)
        if key not in found:
            found[key] = Candidate(ts, key + ".jpg", read)

    for src in sources:
        paths = [src] if src.is_file() else src.rglob("*")
        for p in paths:
            rp = p.resolve()
            if any(rp == e or e in rp.parents for e in excl) or not p.is_file():
                continue
            if p.suffix.lower() == ".jpg" and p.name.startswith("printer_"):
                add(p.name, p.read_bytes)
            elif p.suffix.lower() == ".zip":
                try:
                    z = zipfile.ZipFile(p)
                except zipfile.BadZipFile:
                    continue
                for n in z.namelist():
                    base = n.rsplit("/", 1)[-1]
                    if base.startswith("printer_") and base.lower().endswith(".jpg"):
                        add(base, (lambda z=z, n=n: z.read(n)))
    return found


def labeled_timestamps(raw_dir: Path) -> set[str]:
    """Кадры, которые уже есть в выгрузках Roboflow."""
    done: set[str] = set()
    for z in raw_dir.glob("*.zip"):
        for n in zipfile.ZipFile(z).namelist():
            m = TS_RE.search(n)
            if m:
                done.add(m.group(0))
    return done


def thumbnail(data: bytes) -> np.ndarray | None:
    """Серая миниатюра; None, если кадр не с камеры AD5M."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None or (img.shape[1] * 4, img.shape[0] * 4) != CAMERA_SIZE:
        return None
    return cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA).astype(np.int16)


def select(session: list[Candidate], min_gap: float, max_gap: float, change: float) -> list[Candidate]:
    """Кадры, на которых сцена заметно изменилась (средняя разница миниатюр ≥ change)
    или с прошлого отобранного прошло max_gap секунд; но не чаще min_gap."""
    kept: list[Candidate] = []
    for c in session:
        if c.thumb is None:
            continue
        if not kept:
            kept.append(c)
            continue
        dt = (c.taken - kept[-1].taken).total_seconds()
        if dt < min_gap:
            continue
        diff = float(np.abs(c.thumb - kept[-1].thumb).mean())
        if diff >= change or dt >= max_gap:
            kept.append(c)
    return kept


def cap(kept: list[Candidate], limit: int) -> list[Candidate]:
    """Не больше limit кадров с печати: треть — где модель сильнее всего подозревает проблему,
    остальное — равномерно по времени."""
    if len(kept) <= limit:
        return kept
    by_score = sorted((c for c in kept if c.score > 0), key=lambda c: -c.score)[: limit // 3]
    chosen = {id(c) for c in by_score}
    rest = [c for c in kept if id(c) not in chosen]
    need = limit - len(by_score)
    idx = np.linspace(0, len(rest) - 1, need).round().astype(int)
    picked = by_score + [rest[i] for i in sorted(set(idx))]
    return sorted(picked, key=lambda c: c.taken)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", type=Path, help="папки и zip с кадрами (по умолчанию — вся родительская папка проекта)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-session", type=int, default=120, help="максимум кадров с одной печати")
    ap.add_argument("--min-gap", type=float, default=20, help="не чаще, секунд")
    ap.add_argument("--max-gap", type=float, default=600, help="не реже, секунд")
    ap.add_argument("--change", type=float, default=4.0, help="порог изменения сцены (средняя разница яркости, 0..255)")
    ap.add_argument("--session-gap-min", type=float, default=20)
    ap.add_argument("--include-labeled", action="store_true", help="не пропускать уже размеченные кадры")
    ap.add_argument("--prelabel", action="store_true", help="предразметить текущими моделями models/defects.pt и models/scene.pt")
    ap.add_argument("--conf", type=float, default=0.25, help="порог уверенности для предразметки")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    sources = args.sources or [ROOT.parent]
    if args.out.exists():
        raise SystemExit(f"{args.out} уже существует — укажите новую папку для партии")

    print("Ищу кадры…")
    frames = scan(sources, exclude=[ROOT])
    print(f"  кадров грабера (без повторов): {len(frames)}")
    if not args.include_labeled:
        done = labeled_timestamps(ROOT / "data" / "raw")
        frames = {k: v for k, v in frames.items() if k not in done}
        print(f"  после исключения уже размеченных: {len(frames)}")

    sessions = split_sessions([(c.taken, c) for c in frames.values()], timedelta(minutes=args.session_gap_min))
    print(f"  печатей: {len(sessions)}")

    detector = None
    names = labeling_names()
    if args.prelabel:
        from ddet.detect import Detector
        defects, scene = ROOT / "models" / "defects.pt", ROOT / "models" / "scene.pt"
        if not defects.exists():
            raise SystemExit("Для --prelabel нужна обученная models/defects.pt")
        detector = Detector(defects, scene if scene.exists() else None, device=args.device, min_conf=args.conf)

    out_img, out_lbl = args.out / "images", args.out / "labels"
    out_img.mkdir(parents=True)
    if detector:
        out_lbl.mkdir()
    index = {n: i for i, n in enumerate(names)}
    log = []
    total = 0
    for s_items in sessions:
        session = [c for _, c in s_items]
        for c in session:
            c.thumb = thumbnail(c.read())
        kept = select(session, args.min_gap, args.max_gap, args.change)
        dets_by = {}
        if detector and kept:
            for c in kept:
                img = cv2.imdecode(np.frombuffer(c.read(), np.uint8), cv2.IMREAD_COLOR)
                dets = detector(img)
                dets_by[id(c)] = (img.shape, dets)
                c.score = max((d.conf for d in dets if d.model == "defects"), default=0.0)
        picked = cap(kept, args.per_session)
        for c in picked:
            (out_img / c.name).write_bytes(c.read())
            if detector:
                (h, w, _), dets = dets_by[id(c)]
                lines = [f"{index[d.cls]} {(d.box[0] + d.box[2]) / 2 / w:.6f} {(d.box[1] + d.box[3]) / 2 / h:.6f} "
                         f"{(d.box[2] - d.box[0]) / w:.6f} {(d.box[3] - d.box[1]) / h:.6f}" for d in dets]
                (out_lbl / (Path(c.name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total += len(picked)
        camera = sum(c.thumb is not None for c in session)
        log.append([f"{session[0].taken:%Y-%m-%d %H:%M}", f"{session[-1].taken:%H:%M}", len(session), camera, len(kept), len(picked)])
        print(f"  {log[-1][0]}–{log[-1][1]}: кадров {len(session)}, с камеры AD5M {camera}, изменений {len(kept)}, взято {len(picked)}")
        for c in session:
            c.thumb = None  # не держать миниатюры всех печатей в памяти

    (args.out / "data.yaml").write_text(yaml.safe_dump(
        {"train": "images", "val": "images", "nc": len(names), "names": names}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (args.out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    with open(args.out / "sessions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["начало", "конец", "кадров", "с камеры AD5M", "с изменениями", "взято"])
        w.writerows(log)
    archive = shutil.make_archive(str(args.out), "zip", args.out)
    print(f"\nОтобрано {total} кадров → {args.out}\nАрхив для загрузки в Roboflow/CVAT: {archive}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
