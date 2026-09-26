"""Одни и те же метрики для любых детекторов — по выгруженным предсказаниям.

Формат предсказаний (JSON): {"model": "...", "names": [...], "images": {"file.jpg": [[cls, conf, x1, y1, x2, y2], ...]}}
в пикселях исходного кадра. Разметка — YOLO txt рядом с картинками (../labels).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """AP как площадь под огибающей кривой P(R) (all-point, как в COCO/VOC2010+)."""
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    idx = np.nonzero(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def map50(truth: dict[str, list], preds: dict[str, list], classes: list[int], iou_thr: float = 0.5) -> dict[int, float | None]:
    """truth/preds: файл → [[cls, (conf), x1, y1, x2, y2]]. Возвращает AP50 по классам (None — нет разметки класса)."""
    out: dict[int, float | None] = {}
    for c in classes:
        n_gt = 0
        scored: list[tuple[float, bool]] = []
        for f in truth:
            gt = np.array([t[1:5] for t in truth[f] if t[0] == c], dtype=float).reshape(-1, 4)
            pr = sorted((p for p in preds.get(f, []) if p[0] == c), key=lambda p: -p[1])
            n_gt += len(gt)
            used = np.zeros(len(gt), bool)
            if pr:
                ious = iou_matrix(np.array([p[2:6] for p in pr], float), gt)
                for k, p in enumerate(pr):
                    j = int(np.argmax(ious[k])) if len(gt) else -1
                    hit = j >= 0 and ious[k, j] >= iou_thr and not used[j]
                    if hit:
                        used[j] = True
                    scored.append((p[1], hit))
        if n_gt == 0:
            out[c] = None
            continue
        scored.sort(key=lambda x: -x[0])
        tp = np.cumsum([h for _, h in scored]) if scored else np.array([])
        fp = np.cumsum([not h for _, h in scored]) if scored else np.array([])
        if not len(tp):
            out[c] = 0.0
            continue
        out[c] = average_precision(tp / n_gt, tp / np.maximum(tp + fp, 1e-9))
    return out


def frame_level(truth: dict[str, list], preds: dict[str, list], problems: set[int], thr: float) -> dict[str, float]:
    """Кадр «с проблемой», если в разметке есть проблемный класс; тревога — если модель дала
    проблемный класс с уверенностью ≥ thr. Возвращает recall, долю ложных тревог и точность."""
    tp = fp = pos = neg = 0
    for f in truth:
        t = any(x[0] in problems for x in truth[f])
        p = any(x[0] in problems and x[1] >= thr for x in preds.get(f, []))
        pos += t
        neg += not t
        tp += t and p
        fp += (not t) and p
    return {"recall": tp / max(pos, 1), "false_alarm": fp / max(neg, 1), "precision": tp / max(tp + fp, 1),
            "positives": pos, "negatives": neg}


def per_class_frame_recall(truth: dict[str, list], preds: dict[str, list], thr: float) -> dict[int, tuple[int, int, int]]:
    """По классам: (кадров с классом, найдено, лишних срабатываний)."""
    stats: dict[int, list[int]] = defaultdict(lambda: [0, 0, 0])
    for f in truth:
        t = {x[0] for x in truth[f]}
        p = {x[0] for x in preds.get(f, []) if x[1] >= thr}
        for c in t:
            stats[c][0] += 1
            stats[c][1] += c in p
        for c in p - t:
            stats[c][2] += 1
    return {c: tuple(v) for c, v in stats.items()}
