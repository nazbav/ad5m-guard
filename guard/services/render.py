"""Кадр с рамками распознавания — для панели и снимков тревог."""
from __future__ import annotations

import zlib

import cv2
import numpy as np

from guard.domain.models import Detection

_PALETTE = [(56, 56, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72), (23, 204, 146),
            (134, 219, 61), (211, 188, 0), (209, 99, 0), (255, 56, 132)]
_CONTEXT = {"pei_plate", "glass_plate", "printer_head", "test_line"}


def annotate(img: np.ndarray, dets: list[Detection], defect_conf: float, scene_conf: float,
             header: str, alert: str | None = None, stop: bool = True) -> np.ndarray:
    out = img.copy()
    for d in dets:
        thr = defect_conf if d.model == "defects" else scene_conf
        if d.conf < thr and d.model == "defects":
            continue
        color = (160, 160, 160) if d.cls in _CONTEXT else _PALETTE[zlib.crc32(d.cls.encode()) % len(_PALETTE)]
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1 if d.cls in _CONTEXT else 2)
        if d.cls not in _CONTEXT:
            text = f"{d.cls} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
            cv2.putText(out, text, (x1 + 2, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, header, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if alert:
        h, w = out.shape[:2]
        color = (0, 0, 255) if stop else (0, 140, 255)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, 6)
        cv2.rectangle(out, (0, h - 28), (w, h), color, -1)
        cv2.putText(out, ("STOP: " if stop else "NOTE: ") + alert, (8, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def to_jpg(img: np.ndarray, quality: int = 80) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def from_jpg(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
