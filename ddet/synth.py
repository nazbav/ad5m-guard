"""Искусственные сбои: вырезки настоящего спагетти, вклеенные в нормальные кадры AD5M.

Зачем: настоящих сбоев на камере AD5M — единицы. Вклейка даёт (1) тысячи обучающих
кадров спагетти именно на нашей камере и (2) десятки событий сбоя с точно известным
началом для приёмки. Синтетика помечается везде (имена synth_*, «синтетика» в отчётах):
это не замена настоящим сбоям, а способ получить хоть какую-то меру полноты.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Cutout:
    path: Path
    rgba: np.ndarray            # H×W×4, альфа 0..255
    split: str                  # train | bench — непересекающиеся половины библиотеки


def cutout_split(source_id: str) -> str:
    """Детерминированно делит исходные картинки пополам: одни вырезки — для обучения, другие — для приёмки."""
    return "bench" if int(hashlib.sha1(source_id.encode()).hexdigest(), 16) % 2 else "train"


def make_cutout(img: np.ndarray, mask: np.ndarray, pad: int = 4) -> np.ndarray | None:
    """Вырезка RGBA по маске; None, если маска вырождена."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x2, y2 = min(img.shape[1], xs.max() + pad + 1), min(img.shape[0], ys.max() + pad + 1)
    alpha = (mask[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)                  # мягкий край
    return np.dstack([img[y1:y2, x1:x2], alpha])


def cutout_quality(rgba: np.ndarray) -> tuple[float, float, float]:
    """(заполненность рамки, компактность, плотность контуров) вырезки.

    Спагетти — редкая сеть нитей: маска занимает малую долю рамки, далека от выпуклой
    фигуры и внутри много мелких контуров. Кусок стола, который SAM иногда выделяет
    вместо нити, — почти прямоугольник с ровной текстурой."""
    a = rgba[..., 3] > 127
    area = a.sum()
    if area == 0:
        return 1.0, 1.0, 0.0
    fill = area / a.size
    cnts, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.vstack(cnts))
    solidity = area / max(cv2.contourArea(hull), 1)
    gray = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150) > 0
    inner = cv2.erode(a.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # без границы маски
    density = edges[inner].mean() if inner.any() else 0.0
    return float(fill), float(solidity), float(density)


def good_cutout(rgba: np.ndarray, min_edges: float = 0.08) -> bool:
    fill, solidity, edges = cutout_quality(rgba)
    return fill <= 0.6 and solidity <= 0.85 and edges >= min_edges


def load_library(folder: Path, split: str | None = None, only_good: bool = True) -> list[Cutout]:
    out = []
    for p in sorted(folder.glob("*.png")):
        rgba = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            continue
        if only_good and not good_cutout(rgba):
            continue
        s = p.stem.split("__")[0]
        if split is None or s == split:
            out.append(Cutout(p, rgba, s))
    return out


def paste(img: np.ndarray, cut: np.ndarray, cx: float, cy: float, width: float,
          rng: np.random.Generator) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
    """Вклеивает вырезку в кадр: центр (cx, cy), ширина width пикселей. Возвращает кадр и рамку по маске."""
    h0, w0 = cut.shape[:2]
    scale = width / max(w0, 1)
    w, h = max(4, int(w0 * scale)), max(4, int(h0 * scale))
    part = cv2.resize(cut, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    ix1, iy1 = max(0, x1), max(0, y1)
    ix2, iy2 = min(img.shape[1], x1 + w), min(img.shape[0], y1 + h)
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return img, None
    part = part[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1]
    rgb = part[..., :3].astype(np.float32)
    a = part[..., 3:4].astype(np.float32) / 255.0
    bg = img[iy1:iy2, ix1:ix2].astype(np.float32)
    # Яркость вырезки — к яркости места вклейки (камеры разные, свет разный).
    m = a[..., 0] > 0.5
    if m.sum() > 10:
        gain = np.clip((bg.mean() + 1) / (rgb[m].mean() + 1), 0.6, 1.6) * rng.uniform(0.85, 1.15)
        rgb = np.clip(rgb * gain, 0, 255)
    out = img.copy()
    out[iy1:iy2, ix1:ix2] = (rgb * a + bg * (1 - a)).astype(np.uint8)
    # Мягкость и шум как у камеры AD5M — только в области вклейки.
    region = out[iy1:iy2, ix1:ix2]
    region = cv2.GaussianBlur(region, (3, 3), 0) if rng.random() < 0.7 else region
    noise = rng.normal(0, 3, region.shape)
    out[iy1:iy2, ix1:ix2] = np.clip(region.astype(np.float32) + noise * a, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return out, None
    return out, (float(ix1 + xs.min()), float(iy1 + ys.min()), float(ix1 + xs.max()), float(iy1 + ys.max()))


def distractor(cut: np.ndarray, texture_src: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """«Пустышка»: форма и мягкий край как у вырезки спагетти, а внутри — обычная текстура
    стола из другого кадра. Вклеивается без метки: так модель не может выучить сам след
    вклейки вместо спагетти."""
    h, w = cut.shape[:2]
    H, W = texture_src.shape[:2]
    if h >= H or w >= W:                     # вырезка крупнее кадра — берём весь кадр
        patch = cv2.resize(texture_src, (w, h))
    else:
        x = int(rng.integers(0, W - w + 1))
        y = int(rng.integers(min(H // 3, H - h), H - h + 1))   # по возможности — нижние две трети (стол)
        patch = texture_src[y:y + h, x:x + w]
    return np.dstack([patch, cut[..., 3]])


def median_plate(scene_dataset: Path) -> tuple[float, float, float, float]:
    """Типичная рамка пластины на кадре AD5M 640×480 по разметке сцены — камера у всех AD5M одна."""
    import yaml
    from ddet.labels import parse_text
    names = yaml.safe_load((scene_dataset / "data.yaml").read_text(encoding="utf-8"))["names"]
    plate = [k for k, v in names.items() if v == "pei_plate"][0]
    boxes = [b for p in (scene_dataset / "train" / "labels").glob("own_*.txt")
             for b in parse_text(p.read_text(encoding="utf-8")) if b.cls == plate]
    a = np.array([[(b.xc - b.w / 2) * 640, (b.yc - b.h / 2) * 480, (b.xc + b.w / 2) * 640, (b.yc + b.h / 2) * 480]
                  for b in boxes])
    return tuple(float(v) for v in np.median(a, axis=0))


def growing_event(frames: list, onset_idx: int, duration_s: float, cut: np.ndarray, center: tuple[float, float],
                  width: float, grow_s: float = 180) -> list:
    """Кадры печати с «растущим» комком: с кадра onset_idx комок появляется и за grow_s секунд
    растёт от 40 % до полного размера, держится duration_s. Кадры — [(время, чтение байтов)]."""
    t0 = frames[onset_idx][0]
    out = list(frames)
    for k in range(onset_idx, len(frames)):
        ts, read = frames[k]
        dt = (ts - t0).total_seconds()
        if dt > duration_s:
            break
        w = width * (0.4 + 0.6 * min(1.0, dt / grow_s))

        def composite(read=read, w=w, seed=k):
            img = cv2.imdecode(np.frombuffer(read(), np.uint8), cv2.IMREAD_COLOR)
            img, _ = paste(img, cut, center[0], center[1], w, np.random.default_rng(seed))
            return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        out[k] = (ts, composite)
    return out


def plate_point(plate: tuple[float, float, float, float], rng: np.random.Generator,
                frame: tuple[int, int] = (640, 480)) -> tuple[float, float]:
    """Случайная точка на видимой части пластины: середина рамки стола, не у краёв кадра
    (рамка пластины часто выходит за нижний край, а у переднего края — планка стола)."""
    x1, y1, x2, y2 = plate
    cx1, cy1, cx2, cy2 = max(x1, 40), max(y1, 40), min(x2, frame[0] - 40), min(y2, frame[1] - 60)
    if cx2 - cx1 < 10 or cy2 - cy1 < 10:     # от пластины у края кадра видна узкая полоска — берём её середину
        return (x1 + x2) / 2, (y1 + y2) / 2
    w, h = cx2 - cx1, cy2 - cy1
    return rng.uniform(cx1 + 0.2 * w, cx2 - 0.2 * w), rng.uniform(cy1 + 0.25 * h, cy2 - 0.25 * h)
