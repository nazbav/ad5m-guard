"""Разбор YOLO-разметки: прямоугольники и полигоны приводятся к прямоугольникам."""
from __future__ import annotations

from dataclasses import dataclass

# Рамки уже этого (в долях кадра) считаем мусором разметки.
MIN_SIDE = 1e-3


@dataclass(frozen=True)
class Box:
    cls: int
    xc: float
    yc: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.cls} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def parse_line(line: str) -> Box | None:
    """Строка YOLO → Box.

    Поддерживает оба формата, которые отдаёт Roboflow:
      `cls xc yc w h` — прямоугольник;
      `cls x1 y1 x2 y2 x3 y3 ...` — полигон, берём описанный прямоугольник.
    Координаты обрезаются по границам кадра. Пустая строка и вырожденная
    рамка дают None, битая строка — ValueError.
    """
    parts = line.split()
    if not parts:
        return None
    try:
        cls = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError as e:
        raise ValueError(f"Некорректная строка разметки: {line!r}") from e

    if len(coords) == 4:
        xc, yc, w, h = coords
        x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    elif len(coords) >= 6 and len(coords) % 2 == 0:
        xs, ys = coords[0::2], coords[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        raise ValueError(f"Некорректная строка разметки: {line!r}")

    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    w, h = x2 - x1, y2 - y1
    if w < MIN_SIDE or h < MIN_SIDE:
        return None
    return Box(cls, x1 + w / 2, y1 + h / 2, w, h)


def parse_text(text: str) -> list[Box]:
    return [b for b in (parse_line(l) for l in text.splitlines()) if b is not None]


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.xc + a.w / 2, b.xc + b.w / 2) - max(a.xc - a.w / 2, b.xc - b.w / 2))
    iy = max(0.0, min(a.yc + a.h / 2, b.yc + b.h / 2) - max(a.yc - a.h / 2, b.yc - b.h / 2))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def merge_copies(copies: list[list[Box]], same_iou: float = 0.6, conflict_iou: float = 0.7) -> tuple[list[Box], bool]:
    """Сводит разметку нескольких копий одной картинки.

    Рамки объединяются, почти совпадающие рамки одного класса схлопываются в одну.
    Если одну и ту же область разные копии называют разными классами — это
    противоречие (второй флаг результата), такую картинку надо переразметить.
    """
    merged: list[Box] = []
    for boxes in copies:
        for b in boxes:
            if not any(m.cls == b.cls and iou(m, b) >= same_iou for m in merged):
                merged.append(b)
    conflict = any(a.cls != b.cls and iou(a, b) >= conflict_iou
                   for i, a in enumerate(merged) for b in merged[i + 1:])
    return merged, conflict


def remap(boxes: list[Box], src_names: list[str], mapping: dict[str, str | None],
          dst_names: list[str]) -> list[Box]:
    """Переводит классы из исходной схемы в целевую; класс, отображённый в None, выбрасывается.
    Класс, который уже называется как целевой, переходит сам в себя."""
    dst_index = {n: i for i, n in enumerate(dst_names)}
    out = []
    for b in boxes:
        src = src_names[b.cls]
        if src in mapping:
            dst = mapping[src]
        elif src in dst_index:
            dst = src
        else:
            raise KeyError(f"Класс {src!r} не описан в карте классов")
        if dst is None:
            continue
        out.append(Box(dst_index[dst], b.xc, b.yc, b.w, b.h))
    return out
