"""Деление на train/val/test целыми группами со стратификацией по классам.

Группа — набор почти одинаковых картинок (кадры одного отрезка печати,
дубли одной фотографии из интернета). Если разнести такую группу по разным
выборкам, модель на проверке узнаёт уже виденные кадры и метрики врут.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

# Вес баланса по числу картинок относительно баланса по классам.
IMAGE_WEIGHT = 0.25


@dataclass
class Group:
    id: str
    n_images: int
    classes: Counter = field(default_factory=Counter)  # класс → число объектов


def stratified_group_split(groups: list[Group], ratios: dict[str, float], seed: int = 0,
                           fixed: dict[str, str] | None = None) -> dict[str, str]:
    """Возвращает group.id → имя выборки.

    Жадный алгоритм: группы с самыми редкими классами раскладываются первыми,
    каждая уходит в выборку, которой относительно её доли больше всего не
    хватает объектов этих классов. Поэтому даже класс из трёх групп попадает
    во все три выборки, а не весь в train.

    fixed — группы с заранее известной выборкой (закрепление прошлого деления):
    они учитываются в балансе, остальные раскладываются вокруг них.
    """
    fixed = fixed or {}
    splits = [s for s, r in ratios.items() if r > 0]
    norm = sum(ratios[s] for s in splits)
    share = {s: ratios[s] / norm for s in splits}

    class_total: Counter = Counter()
    for g in groups:
        class_total.update(g.classes)
    n_images = sum(g.n_images for g in groups)

    target_cls = {s: {c: class_total[c] * share[s] for c in class_total} for s in splits}
    need_cls = {s: dict(target_cls[s]) for s in splits}
    target_img = {s: max(n_images * share[s], 1e-9) for s in splits}
    need_img = dict(target_img)

    result: dict[str, str] = {}
    for g in groups:
        s = fixed.get(g.id)
        if s is None:
            continue
        if s not in need_img:
            raise ValueError(f"Закреплённая выборка {s!r} не входит в {splits}")
        result[g.id] = s
        for c, k in g.classes.items():
            need_cls[s][c] -= k
        need_img[s] -= g.n_images

    order = [g for g in groups if g.id not in result]
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: (
        min((class_total[c] for c in g.classes), default=math.inf),
        -sum(g.classes.values()),
        -g.n_images,
    ))

    for g in order:
        def score(s: str) -> tuple[float, float]:
            img_score = need_img[s] / target_img[s]
            if not g.classes:
                return img_score, share[s]
            weights = {c: g.classes[c] / class_total[c] for c in g.classes}
            cls_score = sum(w * need_cls[s][c] / max(target_cls[s][c], 1e-9)
                            for c, w in weights.items()) / sum(weights.values())
            return cls_score + IMAGE_WEIGHT * img_score, share[s]

        best = max(splits, key=score)
        result[g.id] = best
        for c, k in g.classes.items():
            need_cls[best][c] -= k
        need_img[best] -= g.n_images
    return result
