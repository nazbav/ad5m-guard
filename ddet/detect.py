"""Запуск моделей брака и сцены на кадре и отрисовка результата."""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    model: str                                  # "defects" или "scene"
    cls: str
    conf: float
    box: tuple[float, float, float, float]      # x1, y1, x2, y2 в пикселях


class YoloBackend:
    """Модель Ultralytics YOLO (.pt)."""

    def __init__(self, path: Path, device: str | None, imgsz: int):
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        self.names = self.model.names
        self.ckpt_path = str(path)
        self.device, self.imgsz = device, imgsz

    def predict(self, images: list[np.ndarray], conf: float) -> list[list[tuple[int, float, tuple]]]:
        results = self.model.predict(images, conf=conf, imgsz=self.imgsz, device=self.device, verbose=False)
        return [[(int(c), float(s), tuple(b)) for c, s, b in
                 zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())] for r in results]


class RFDETRBackend:
    """Модель RF-DETR (.pth, лицензия Apache-2.0). Размер берётся из имени файла: *_rfdetr_small.pth."""

    SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium"}

    def __init__(self, path: Path, device: str | None):
        import rfdetr
        size = next((s for s in self.SIZES if f"rfdetr_{s}" in Path(path).stem or f"_{s}" in Path(path).parent.name), "small")
        kwargs = {"pretrain_weights": str(path)}
        if device is not None:
            kwargs["device"] = "cpu" if str(device) == "cpu" else "cuda"
        self.model = getattr(rfdetr, self.SIZES[size])(**kwargs)
        names = self.model.class_names
        self.names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
        self.ckpt_path = str(path)

    def predict(self, images: list[np.ndarray], conf: float) -> list[list[tuple[int, float, tuple]]]:
        from PIL import Image
        rgb = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in images]
        dets = self.model.predict(rgb if len(rgb) > 1 else rgb[0], threshold=conf)
        dets = dets if isinstance(dets, list) else [dets]
        return [[(int(c), float(s), tuple(b)) for c, s, b in
                 zip(d.class_id.tolist(), d.confidence.tolist(), d.xyxy.tolist())] for d in dets]


class OnnxBackend:
    """ONNX-модель через адаптер приложения — то, что реально поедет в exe."""

    def __init__(self, path: Path):
        from guard.adapters.onnx_detector import OnnxModel
        self.model = OnnxModel(path)
        self.names = self.model.names
        self.ckpt_path = str(path)

    def predict(self, images: list[np.ndarray], conf: float) -> list[list[tuple[int, float, tuple]]]:
        return [self.model.predict(im, conf) for im in images]


def load_backend(path: Path, device: str | None, imgsz: int = 640):
    suffix = Path(path).suffix
    if suffix == ".onnx":
        return OnnxBackend(path)
    return RFDETRBackend(path, device) if suffix == ".pth" else YoloBackend(path, device, imgsz)


class Detector:
    """Модель брака и (необязательно) модель сцены: YOLO (.pt) или RF-DETR (.pth)."""

    def __init__(self, defects: Path, scene: Path | None = None, device: str | None = None,
                 min_conf: float = 0.2, imgsz: int = 640):
        self.models = {"defects": load_backend(defects, device, imgsz)}
        if scene:
            self.models["scene"] = load_backend(scene, device, imgsz)
        self.min_conf = min_conf

    def __call__(self, image: np.ndarray) -> list[Detection]:
        return self.batch([image])[0]

    def batch(self, images: list[np.ndarray], batch_size: int = 16) -> list[list[Detection]]:
        """Кадры со многих принтеров одним проходом — так один GPU тянет сотни камер."""
        out: list[list[Detection]] = [[] for _ in images]
        for name, model in self.models.items():
            for start in range(0, len(images), batch_size):
                chunk = images[start:start + batch_size]
                for k, dets in enumerate(model.predict(chunk, self.min_conf)):
                    out[start + k] += [Detection(name, model.names[c], s, b) for c, s, b in dets]
        return out


_PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
            (10, 249, 72), (23, 204, 146), (134, 219, 61), (211, 188, 0), (209, 99, 0)]


def draw(image: np.ndarray, dets: list[Detection], thresholds: dict[str, float], header: str,
         alert: str | None = None, kind: str = "STOP") -> np.ndarray:
    """Рамки выше порога — сплошные, ниже — тонкие; сверху строка состояния,
    при тревоге — рамка кадра: красная для остановки (STOP), оранжевая для уведомления (NOTE)."""
    img = image.copy()
    for d in dets:
        color = _PALETTE[zlib.crc32(d.cls.encode()) % len(_PALETTE)]  # hash() меняется между запусками
        strong = d.conf >= thresholds.get(d.model, 0.4)
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if strong else 1)
        if strong:
            text = f"{d.cls} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
            cv2.putText(img, text, (x1 + 2, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, header, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if alert:
        h, w = img.shape[:2]
        color = (0, 0, 255) if kind == "STOP" else (0, 140, 255)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 6)
        cv2.rectangle(img, (0, h - 28), (w, h), color, -1)
        cv2.putText(img, f"{kind}: {alert}", (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img
