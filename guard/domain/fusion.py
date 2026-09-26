"""Две модели брака на одном кадре → один список детекций «defects». Чистая логика.

    and — объект засчитан, только если обе модели нашли его на одном месте; уверенность — меньшая
    or  — любая из моделей
    avg — средняя уверенность; если вторая модель объект не нашла, её уверенность считается нулём
"""
from __future__ import annotations

from guard.domain.models import Detection

MODES = ("and", "or", "avg")
# «То же место»: меньшая рамка хотя бы наполовину внутри большей. IoU здесь не годится — на коме нити
# одна модель обводит кусок, другая весь ком, и IoU у совпадающих находок бывает 0.1.
MATCH_OVERLAP = 0.5


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def overlap(a, b) -> float:
    """Доля площади меньшей из рамок, лежащая внутри другой."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / max(small, 1e-9)


def fuse_defects(a: list[Detection], b: list[Detection], mode: str) -> list[Detection]:
    if mode not in MODES:
        raise ValueError(f"способ объединения моделей: {MODES}")
    if mode == "or":
        return [Detection("defects", d.cls, d.conf, d.box) for d in a + b]
    free = sorted(b, key=lambda d: -d.conf)
    out = []
    for d in sorted(a, key=lambda d: -d.conf):
        # из совпадающих по месту — самая уверенная находка второй модели
        same = [x for x in free if x.cls == d.cls and overlap(d.box, x.box) >= MATCH_OVERLAP]
        m = max(same, key=lambda x: x.conf, default=None)
        if m is not None:
            free.remove(m)
            conf = min(d.conf, m.conf) if mode == "and" else (d.conf + m.conf) / 2
            out.append(Detection("defects", d.cls, conf, d.box))
        elif mode == "avg":
            out.append(Detection("defects", d.cls, d.conf / 2, d.box))
    if mode == "avg":
        out += [Detection("defects", x.cls, x.conf / 2, x.box) for x in free]
    return out
