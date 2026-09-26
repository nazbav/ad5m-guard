"""Полуавтоматическая разметка партии кадров: кандидаты моделей → листы на просмотр → решения → разметка.

    python tools/review.py candidates data/to_label/batch1        # прогнать модели, записать кандидатов
    python tools/review.py sheets data/to_label/batch1            # листы 3×3 для просмотра
    python tools/review.py apply data/to_label/batch1 --out data/labeled/batch1

Решения — текстовые файлы review/decisions/*.txt внутри партии, строка на кадр листа:
    <номер на листе> <токены…>
Токены:
    .                   брака нет (кандидаты брака отбрасываются)
    a b                 оставить кандидатов с этими буквами как есть
    a=garbage           оставить кандидата, но с другим классом
    +класс@x,y          добавить объект: рамку по точке строит SAM
    +класс@x1,y1,x2,y2  добавить объект заданной рамкой
    glass               на столе стекло вместо PEI (пластина размечается как glass_plate)
    nocam               стол не виден (камера сдвинута/закрыта): пластина не размечается
    noplate             пластины на столе нет
    misplaced           пластина лежит криво: взять рамку pei_misplaced у модели сцены
    nocover             с головы снят кожух обдува: голова размечается как head_no_cover
    skip                кадр плохой, в разметку не брать
Координаты — пиксели исходного кадра 640×480 (на листах есть сетка через 80 px).
Стол, голова, тестовая линия и рука берутся из модели сцены (уверенность ≥ 0.5).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.config import CONFIG_DIR, ROOT, labeling_names, load_classes  # noqa: E402
from ddet.frames import parse_timestamp, split_sessions  # noqa: E402

# Порог авторазметки сцены. Пластину модель находит почти безошибочно (mAP50 0.995), а кадры
# без пластины разметчики помечают флагами noplate/nocam — поэтому для неё порог ниже.
SCENE_AUTO = {"pei_plate": 0.3, "printer_head": 0.5, "test_line": 0.5, "hand": 0.5}
CANDIDATE_CONF = 0.15
MAX_CANDIDATES = 8
TILE = (426, 320)
GRID = 3
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def review_dir(batch: Path) -> Path:
    return batch / "review"


# ------------------------------------------------------------------ кандидаты
def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe(dets: list, iou: float = 0.4) -> list:
    """YOLO26 работает без NMS, и на низком пороге один объект даёт несколько рамок —
    на листе оставляем одну, самую уверенную (без учёта класса)."""
    kept = []
    for d in dets:
        if all(_iou(d.box, k.box) < iou for k in kept):
            kept.append(d)
    return kept


def cmd_candidates(batch: Path, device: str) -> None:
    from ddet.detect import Detector
    det = Detector(ROOT / "models" / "defects.pt", ROOT / "models" / "scene.pt", device=device, min_conf=CANDIDATE_CONF)
    images = sorted((batch / "images").glob("*.jpg"))
    out: dict[str, dict] = {}
    for i in range(0, len(images), 16):
        chunk = images[i:i + 16]
        imgs = [cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR) for p in chunk]
        for p, img, dets in zip(chunk, imgs, det.batch(imgs)):
            defects = dedupe(sorted((d for d in dets if d.model == "defects"), key=lambda d: -d.conf))[:MAX_CANDIDATES]
            out[p.name] = {
                "size": [img.shape[1], img.shape[0]],
                "defects": [{"id": LETTERS[k], "cls": d.cls, "conf": round(d.conf, 3), "box": [round(v, 1) for v in d.box]}
                            for k, d in enumerate(defects)],
                "scene": [{"cls": d.cls, "conf": round(d.conf, 3), "box": [round(v, 1) for v in d.box]}
                          for d in dets if d.model == "scene"],
            }
        print(f"  {min(i + 16, len(images))}/{len(images)}")
    review_dir(batch).mkdir(exist_ok=True)
    (review_dir(batch) / "candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"Кандидаты: {review_dir(batch) / 'candidates.json'}")


# ------------------------------------------------------------------ листы
_ABBR = {"spaghetti": "SPAG", "stringing": "STR", "warping": "WARP", "cracks": "CRK", "garbage": "GARB", "detached": "DET"}


def ordered_frames(batch: Path) -> list[list[str]]:
    """Кадры по печатям, внутри печати — по времени."""
    names = [p.name for p in (batch / "images").glob("*.jpg")]
    sessions = split_sessions([(parse_timestamp(n), n) for n in names], timedelta(minutes=20))
    return [[n for _, n in s] for s in sessions]


def cmd_zoom(batch: Path, sheet: str, numbers: list[int]) -> None:
    """Кадры листа в полном разрешении ×2 — для спорных случаев."""
    cands = json.loads((review_dir(batch) / "candidates.json").read_text(encoding="utf-8"))
    frames = json.loads((review_dir(batch) / "sheets.json").read_text(encoding="utf-8"))[sheet]["frames"]
    out = review_dir(batch) / "zoom"
    out.mkdir(exist_ok=True)
    for n in numbers:
        name = frames[n - 1]
        img = cv2.imdecode(np.fromfile(str(batch / "images" / name), np.uint8), cv2.IMREAD_COLOR)
        big = draw_tile(img, cands[name], n, name, size=(img.shape[1] * 2, img.shape[0] * 2))
        path = out / f"{sheet}_{n}.jpg"
        cv2.imencode(".jpg", big, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tofile(str(path))
        print(path)


def draw_tile(img: np.ndarray, cand: dict, number: int, label: str, size: tuple[int, int] = TILE) -> np.ndarray:
    h, w = img.shape[:2]
    t = img.copy()
    for x in range(80, w, 80):
        cv2.line(t, (x, 0), (x, h), (200, 200, 200), 1)
        cv2.putText(t, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    for y in range(80, h, 80):
        cv2.line(t, (0, y), (w, y), (200, 200, 200), 1)
        cv2.putText(t, str(y), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    img = cv2.addWeighted(t, 0.35, img, 0.65, 0)
    for c in cand["defects"]:
        x1, y1, x2, y2 = map(int, c["box"])
        color = (0, 0, 255) if c["conf"] >= 0.4 else (0, 200, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # крупная буква слева от рамки, мелкая подпись класса под ней
        lx, ly = max(0, x1 - 22), max(26, y1 + 20)
        cv2.putText(img, c["id"], (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(img, c["id"], (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        text = f"{_ABBR.get(c['cls'], c['cls'])}{int(c['conf'] * 100)}"
        cv2.putText(img, text, (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, text, (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for c in cand["scene"]:
        if c["cls"] in ("pei_misplaced", "hand") and c["conf"] >= 0.4:
            x1, y1, x2, y2 = map(int, c["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 160, 0), 1)
            cv2.putText(img, f"{c['cls']} {c['conf']:.2f}", (x1, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 160, 0), 1)
    plate = max((c["conf"] for c in cand["scene"] if c["cls"] == "pei_plate"), default=0)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA if size[0] < img.shape[1] else cv2.INTER_CUBIC)
    cv2.rectangle(img, (0, 0), (size[0], 22), (0, 0, 0), -1)
    cv2.putText(img, f"{number}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, f"{label}  plate {plate:.2f}", (34, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def cmd_sheets(batch: Path) -> None:
    cands = json.loads((review_dir(batch) / "candidates.json").read_text(encoding="utf-8"))
    sheets_dir = review_dir(batch) / "sheets"
    if sheets_dir.exists():
        shutil.rmtree(sheets_dir)
    sheets_dir.mkdir(parents=True)
    index = {}
    k = 0
    per = GRID * GRID
    for s_idx, session in enumerate(ordered_frames(batch)):
        for start in range(0, len(session), per):
            k += 1
            names = session[start:start + per]
            sheet = np.zeros((TILE[1] * GRID, TILE[0] * GRID, 3), np.uint8)
            for j, n in enumerate(names):
                img = cv2.imdecode(np.fromfile(str(batch / "images" / n), np.uint8), cv2.IMREAD_COLOR)
                ts = parse_timestamp(n)
                tile = draw_tile(img, cands[n], j + 1, f"{ts:%m-%d %H:%M:%S}")
                r, c = divmod(j, GRID)
                sheet[r * TILE[1]:(r + 1) * TILE[1], c * TILE[0]:(c + 1) * TILE[0]] = tile
            name = f"{k:04d}"
            cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(str(sheets_dir / f"{name}.jpg"))
            index[name] = {"session": s_idx, "frames": names}
    (review_dir(batch) / "sheets.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Листов: {k} → {sheets_dir}")


# ------------------------------------------------------------------ решения
@dataclass
class Decision:
    keep: dict[str, str | None] = field(default_factory=dict)   # буква → новый класс (None — как было)
    add: list[tuple[str, list[float]]] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    clean: bool = False


TOKEN_ADD = re.compile(r"^\+(\w+)@([\d.]+(?:,[\d.]+){1,3})$")
FLAGS = {"glass", "nocam", "noplate", "misplaced", "nocover", "skip"}


def parse_decisions(text: str, sheet_frames: dict[str, list[str]], defect_names: set[str]) -> dict[str, Decision]:
    """Разбирает файл решений. Заголовок листа — строка «# NNNN»."""
    result: dict[str, Decision] = {}
    sheet = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("//")[0].strip()
        if not line:
            continue
        m = re.match(r"^#\s*(\d{4})", line)
        if m:
            sheet = m.group(1)
            if sheet not in sheet_frames:
                raise ValueError(f"строка {lineno}: нет листа {sheet}")
            continue
        if sheet is None:
            raise ValueError(f"строка {lineno}: решение до заголовка листа")
        parts = line.split()
        idx = int(parts[0]) - 1
        frames = sheet_frames[sheet]
        if not 0 <= idx < len(frames):
            raise ValueError(f"строка {lineno}: на листе {sheet} нет кадра {idx + 1}")
        d = Decision()
        for tok in parts[1:]:
            if tok == ".":
                d.clean = True
            elif tok in FLAGS:
                d.flags.add(tok)
            elif (m := TOKEN_ADD.match(tok)):
                cls = m.group(1)
                if cls not in defect_names | {"pei_misplaced", "head_no_cover", "hand", "glass_plate", "pei_plate"}:
                    raise ValueError(f"строка {lineno}: неизвестный класс {cls}")
                d.add.append((cls, [float(v) for v in m.group(2).split(",")]))
            elif re.fullmatch(r"[a-z](=\w+)?", tok):
                letter, _, new = tok.partition("=")
                if new and new not in defect_names:
                    raise ValueError(f"строка {lineno}: неизвестный класс {new}")
                d.keep[letter] = new or None
            else:
                raise ValueError(f"строка {lineno}: не понимаю «{tok}»")
        result[frames[idx]] = d
    return result


class Refiner:
    """Точка → рамка через SAM; без SAM — квадрат вокруг точки."""

    def __init__(self, device: str):
        from ultralytics import SAM
        self.sam = SAM(str(ROOT / "weights" / "mobile_sam.pt"))
        self.device = device

    def box(self, img: np.ndarray, x: float, y: float) -> list[float]:
        h, w = img.shape[:2]
        res = self.sam(img, points=[[x, y]], labels=[1], device=self.device, verbose=False)[0]
        if res.masks is not None and len(res.masks.data):
            m = res.masks.data[0].cpu().numpy() > 0.5
            ys, xs = np.nonzero(m)
            if len(xs):
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                area = (x2 - x1) * (y2 - y1)
                if 16 <= area <= 0.25 * w * h:  # SAM иногда выделяет весь стол — тогда не верим
                    return [float(x1), float(y1), float(x2), float(y2)]
        return [max(0, x - 20), max(0, y - 20), min(w, x + 20), min(h, y + 20)]


def yolo_line(idx: int, box: list[float], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    return f"{idx} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def cmd_check(batch: Path, file: Path) -> None:
    """Проверить один файл решений: формат, полнота листов, существование букв."""
    cands = json.loads((review_dir(batch) / "candidates.json").read_text(encoding="utf-8"))
    sheets = json.loads((review_dir(batch) / "sheets.json").read_text(encoding="utf-8"))
    sheet_frames = {k: v["frames"] for k, v in sheets.items()}
    defect_names = set(load_classes(CONFIG_DIR / "defects.yaml").names)
    try:
        decisions = parse_decisions(file.read_text(encoding="utf-8"), sheet_frames, defect_names)
    except ValueError as e:
        raise SystemExit(f"ОШИБКА: {e}")
    errors = []
    touched = {k for k, fr in sheet_frames.items() if any(n in decisions for n in fr)}
    for k in sorted(touched):
        missing = [i + 1 for i, n in enumerate(sheet_frames[k]) if n not in decisions]
        if missing:
            errors.append(f"лист {k}: нет решений для кадров {missing}")
    for frame, d in decisions.items():
        ids = {c["id"] for c in cands[frame]["defects"]}
        bad = set(d.keep) - ids
        if bad:
            errors.append(f"{frame}: нет кандидатов {sorted(bad)}")
        if "misplaced" in d.flags and not any(s["cls"] == "pei_misplaced" for s in cands[frame]["scene"]):
            errors.append(f"{frame}: misplaced, но у модели нет рамки — задайте +pei_misplaced@x1,y1,x2,y2")
        if not (d.clean or d.keep or d.add or d.flags):
            errors.append(f"{frame}: пустое решение")
    if errors:
        raise SystemExit("ОШИБКИ:\n" + "\n".join(errors))
    counts = {}
    for d in decisions.values():
        for cls in list(d.keep.values()) + [a[0] for a in d.add]:
            counts[cls or "как у модели"] = counts.get(cls or "как у модели", 0) + 1
    print(f"OK: листов {len(touched)}, кадров {len(decisions)}; оставлено кандидатов {sum(len(d.keep) for d in decisions.values())}, "
          f"добавлено {sum(len(d.add) for d in decisions.values())}, skip {sum('skip' in d.flags for d in decisions.values())}")


def dedupe_lines(lines: list[str], w: int, h: int, iou: float = 0.5) -> list[str]:
    """Рамки одного класса, почти совпадающие (добавлена руками и пришла от модели), — оставить первую."""
    kept: list[tuple[str, list[float]]] = []
    out = []
    for line in lines:
        cls, xc, yc, bw, bh = line.split()
        box = [float(xc) - float(bw) / 2, float(yc) - float(bh) / 2, float(xc) + float(bw) / 2, float(yc) + float(bh) / 2]
        if any(c == cls and _iou(box, b) >= iou for c, b in kept):
            continue
        kept.append((cls, box))
        out.append(line)
    return out


def cmd_apply(batch: Path, out: Path, device: str, allow_partial: bool) -> None:
    cands = json.loads((review_dir(batch) / "candidates.json").read_text(encoding="utf-8"))
    sheets = json.loads((review_dir(batch) / "sheets.json").read_text(encoding="utf-8"))
    sheet_frames = {k: v["frames"] for k, v in sheets.items()}
    defect_names = set(load_classes(CONFIG_DIR / "defects.yaml").names)
    decisions: dict[str, Decision] = {}
    for f in sorted((review_dir(batch) / "decisions").glob("*.txt")):
        decisions.update(parse_decisions(f.read_text(encoding="utf-8"), sheet_frames, defect_names))
    reviewed_sheets = {k for k, fr in sheet_frames.items() if all(n in decisions for n in fr)}
    missing = [k for k, fr in sheet_frames.items() if any(n in decisions for n in fr) and k not in reviewed_sheets]
    if missing:
        raise SystemExit(f"На листах {missing} решения не по всем кадрам")
    if not allow_partial and len(reviewed_sheets) < len(sheet_frames):
        raise SystemExit(f"Просмотрено {len(reviewed_sheets)} из {len(sheet_frames)} листов (--partial — взять готовые)")

    names = labeling_names()
    index = {n: i for i, n in enumerate(names)}
    refiner = None
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir()
    stats = {n: 0 for n in names}
    frames = 0
    for frame, d in decisions.items():
        if "skip" in d.flags:
            continue
        c = cands[frame]
        w, h = c["size"]
        lines = []
        by_id = {x["id"]: x for x in c["defects"]}
        for letter, new in d.keep.items():
            if letter not in by_id:
                raise SystemExit(f"{frame}: нет кандидата «{letter}»")
            cls = new or by_id[letter]["cls"]
            lines.append(yolo_line(index[cls], by_id[letter]["box"], w, h))
            stats[cls] += 1
        if d.add:
            img = cv2.imdecode(np.fromfile(str(batch / "images" / frame), np.uint8), cv2.IMREAD_COLOR)
            for cls, v in d.add:
                if len(v) == 4:
                    box = v
                else:
                    refiner = refiner or Refiner(device)
                    box = refiner.box(img, v[0], v[1])
                lines.append(yolo_line(index[cls], box, w, h))
                stats[cls] += 1
        if "misplaced" in d.flags:
            mis = max((s for s in c["scene"] if s["cls"] == "pei_misplaced"), key=lambda s: s["conf"], default=None)
            if mis is None:
                raise SystemExit(f"{frame}: флаг misplaced, но модель сцены не дала рамку — задайте +pei_misplaced@x1,y1,x2,y2")
            lines.append(yolo_line(index["pei_misplaced"], mis["box"], w, h))
            stats["pei_misplaced"] += 1
        for s in c["scene"]:
            cls = s["cls"]
            if cls not in SCENE_AUTO or s["conf"] < SCENE_AUTO[cls]:
                continue
            if cls == "pei_plate":
                if d.flags & {"nocam", "noplate"}:
                    continue
                if "glass" in d.flags:
                    cls = "glass_plate"
            if cls == "printer_head" and "nocover" in d.flags:
                cls = "head_no_cover"
            lines.append(yolo_line(index[cls], s["box"], w, h))
            stats[cls] += 1
        lines = dedupe_lines(lines, w, h)
        shutil.copy2(batch / "images" / frame, out / "images" / frame)
        (out / "labels" / (Path(frame).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        frames += 1
    (out / "data.yaml").write_text(yaml.safe_dump({"train": "images", "val": "images", "nc": len(names), "names": names},
                                                  allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Размечено кадров: {frames} (листов {len(reviewed_sheets)} из {len(sheet_frames)}) → {out}")
    for n, k in stats.items():
        if k:
            print(f"  {n}: {k}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("candidates", "sheets", "zoom", "check", "apply"))
    ap.add_argument("batch", type=Path)
    ap.add_argument("sheet", nargs="?", help="zoom: номер листа, например 0003; check: файл решений")
    ap.add_argument("numbers", nargs="*", type=int, help="zoom: номера кадров на листе")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--partial", action="store_true", help="apply: взять только просмотренные листы")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    if args.command == "candidates":
        cmd_candidates(args.batch, args.device)
    elif args.command == "sheets":
        cmd_sheets(args.batch)
    elif args.command == "zoom":
        cmd_zoom(args.batch, args.sheet, args.numbers)
    elif args.command == "check":
        cmd_check(args.batch, Path(args.sheet))
    else:
        cmd_apply(args.batch, args.out or ROOT / "data" / "labeled" / args.batch.name, args.device, args.partial)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
