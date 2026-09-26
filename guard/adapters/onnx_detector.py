"""Распознавание на ONNX Runtime (без PyTorch): модель брака + модель сцены.

Формат модели — выгрузка Ultralytics YOLO в ONNX: вход 1×3×640×640 RGB 0..1 с letterbox,
выход 1×(4+классов)×N (cx, cy, w, h, уверенности классов). Имена классов — в метаданных модели.
Постобработка: порог → NMS по классам (как у Ultralytics, IoU 0.7) → координаты исходного кадра.
"""
from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np

from guard.domain.models import Detection

NMS_IOU = 0.7
MAX_DET = 100


def letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Вписать кадр во вход модели size = (высота, ширина) с серыми полями по центру."""
    th, tw = size
    h, w = img.shape[:2]
    r = min(th / h, tw / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (tw - nw) / 2, (th - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


class OnnxModel:
    def __init__(self, path: Path, providers: list[str] | None = None, threads: int = 0):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), opts, providers=providers or ["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names: dict[int, str] = ast.literal_eval(meta["names"])
        imgsz = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
        self.size = (int(imgsz[0]), int(imgsz[1])) if isinstance(imgsz, (list, tuple)) else (int(imgsz), int(imgsz))
        self.path = Path(path)
        self.arch = meta.get("arch", "yolo")
        if self.arch == "rfdetr":
            self.outputs = [o.name for o in self.session.get_outputs()]
            self.mean = np.array(ast.literal_eval(meta["mean"]), np.float32)
            self.std = np.array(ast.literal_eval(meta["std"]), np.float32)

    def predict(self, img: np.ndarray, conf: float) -> list[tuple[int, float, tuple[float, float, float, float]]]:
        if self.arch == "rfdetr":
            return self._predict_detr(img, conf)
        lb, r, (px, py) = letterbox(img, self.size)
        x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(None, {self.input: x})[0][0]
        h, w = img.shape[:2]
        if out.ndim == 2 and out.shape[1] == 6:                        # голова без NMS (YOLO26 end2end): x1 y1 x2 y2 conf cls
            res = []
            for x1, y1, x2, y2, s, c in out:
                if s < conf:
                    continue
                x1, y1, x2, y2 = (np.array([x1, y1, x2, y2]) - [px, py, px, py]) / r
                res.append((int(c), float(s), (float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)),
                                               float(np.clip(x2, 0, w)), float(np.clip(y2, 0, h)))))
            return sorted(res, key=lambda d: -d[1])[:MAX_DET]
        if out.shape[0] < out.shape[1] and out.shape[0] == 4 + len(self.names):
            out = out.T                                               # (4+nc, N) → (N, 4+nc)
        scores = out[:, 4:]
        cls = scores.argmax(1)
        best = scores[np.arange(len(cls)), cls]
        keep = best >= conf
        if not keep.any():
            return []
        b, cls, best = out[keep, :4], cls[keep], best[keep]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        # NMS по классам: сдвиг рамок разных классов, чтобы они не подавляли друг друга
        offset = cls[:, None].astype(np.float32) * 4096
        order = self._nms(xyxy + offset, best)[:MAX_DET]
        res = []
        for i in order:
            x1, y1, x2, y2 = (xyxy[i] - [px, py, px, py]) / r
            res.append((int(cls[i]), float(best[i]),
                        (float(np.clip(x1, 0, w)), float(np.clip(y1, 0, h)), float(np.clip(x2, 0, w)), float(np.clip(y2, 0, h)))))
        return res

    def _predict_detr(self, img: np.ndarray, conf: float) -> list[tuple[int, float, tuple[float, float, float, float]]]:
        """RF-DETR: кадр растягивается в квадрат входа (как при обучении), RGB, нормализация ImageNet;
        выход — рамки cx cy w h в долях кадра и логиты классов; лучшие пары запрос×класс (NMS не нужен)."""
        h, w = img.shape[:2]
        x = cv2.cvtColor(cv2.resize(img, (self.size[1], self.size[0]), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        x = ((x.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)[None]
        outs = dict(zip(self.outputs, self.session.run(None, {self.input: np.ascontiguousarray(x)})))
        boxes, prob = outs["dets"][0], 1.0 / (1.0 + np.exp(-outs["labels"][0]))
        flat = prob.ravel()
        idx = np.flatnonzero(flat >= conf)
        idx = idx[np.argsort(-flat[idx])][:MAX_DET]
        res = []
        for i in idx:
            q, c = divmod(int(i), prob.shape[1])
            if c not in self.names:                                   # служебный «фон»
                continue
            cx, cy, bw, bh = boxes[q]
            res.append((c, float(flat[i]), (float(np.clip((cx - bw / 2) * w, 0, w)), float(np.clip((cy - bh / 2) * h, 0, h)),
                                            float(np.clip((cx + bw / 2) * w, 0, w)), float(np.clip((cy + bh / 2) * h, 0, h)))))
        return res

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        order = scores.argsort()[::-1]
        keep = []
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        while order.size:
            i = order[0]
            keep.append(int(i))
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            order = rest[inter / (area[i] + area[rest] - inter + 1e-9) <= NMS_IOU]
        return keep


class OnnxDetector:
    """Модель брака (+ необязательная вторая модель брака, объединение по guard.domain.fusion) + модель сцены.

    В режиме «and» вторая модель запускается только на кадрах, где первая что-то нашла не ниже
    gate_conf, — на чистой печати она почти не работает и не отнимает процессор."""

    def __init__(self, defects: Path, scene: Path | None = None, min_conf: float = 0.15,
                 providers: list[str] | None = None, second: Path | None = None, fusion: str = "and",
                 gate_conf: float = 0.3):
        from guard.domain.fusion import MODES
        if fusion not in MODES:
            raise ValueError(f"способ объединения моделей: {MODES}")
        self.models = {"defects": OnnxModel(defects, providers)}
        if second:
            self.models["defects2"] = OnnxModel(second, providers)
        if scene and Path(scene).exists():
            self.models["scene"] = OnnxModel(scene, providers)
        self.min_conf, self.fusion, self.gate_conf = min_conf, fusion, gate_conf
        self.second_runs = 0
        self.focus: set[str] | None = None      # классы, которые что-то решают (задаёт сервис по настройкам)

    @property
    def has_scene(self) -> bool:
        return "scene" in self.models

    def _run(self, name: str, img: np.ndarray) -> list[Detection]:
        m = self.models[name]
        label = "defects" if name.startswith("defects") else name
        return [Detection(label, m.names[c], s, b) for c, s, b in m.predict(img, self.min_conf)]

    def detect(self, images: list[np.ndarray]) -> list[list[Detection]]:
        from guard.domain.fusion import fuse_defects
        out = []
        for img in images:
            first = self._run("defects", img)
            if "defects2" in self.models:
                # Объединяются только важные классы; остальное первой модели — как есть (только подпись на кадре).
                key = [d for d in first if self.focus is None or d.cls in self.focus]
                rest = [d for d in first if not (self.focus is None or d.cls in self.focus)]
                if self.fusion != "and" or any(d.conf >= self.gate_conf for d in key):
                    self.second_runs += 1
                    second = [d for d in self._run("defects2", img) if self.focus is None or d.cls in self.focus]
                    key = fuse_defects(key, second, self.fusion)
                else:
                    key = []                        # «and»: первая ничего уверенного не нашла — объединять нечего
                first = key + rest
            out.append(first + (self._run("scene", img) if self.has_scene else []))
        return out
