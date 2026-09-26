"""Источники кадров: видеофайл, папка или zip с кадрами грабера, живой поток камеры."""
from __future__ import annotations

import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from ddet.frames import parse_timestamp

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}


@dataclass
class Frame:
    index: int          # номер кадра в источнике
    t: float            # секунды от начала: реальное время для кадров грабера и потока, время видео для файла
    image: np.ndarray   # BGR
    label: str          # подпись для отчёта: время съёмки или позиция в видео


def decode(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def _fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _video(path: Path, every: float) -> Iterator[Frame]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Не удалось открыть видео {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(every * fps))
    i = 0
    try:
        while True:
            if i % step:
                if not cap.grab():
                    break
            else:
                ok, img = cap.read()
                if not ok:
                    break
                yield Frame(i, i / fps, img, _fmt(i / fps))
            i += 1
    finally:
        cap.release()


def _sequence(entries: list[tuple[str, callable]], every: float) -> Iterator[Frame]:
    """Кадры грабера. Порядок и время — по времени съёмки из имени файла, если оно есть."""
    stamped = [(parse_timestamp(Path(n).name), n, read) for n, read in entries]
    if stamped and all(ts is not None for ts, _, _ in stamped):
        stamped.sort(key=lambda x: x[0])
        t0 = stamped[0][0]
        times = [(ts - t0).total_seconds() for ts, _, _ in stamped]
        labels = [f"{ts:%Y-%m-%d %H:%M:%S}" for ts, _, _ in stamped]
    else:
        stamped.sort(key=lambda x: x[1])
        times = [float(i) for i in range(len(stamped))]
        labels = [Path(n).name for _, n, _ in stamped]
    last = None
    for i, ((_, name, read), t, label) in enumerate(zip(stamped, times, labels)):
        if last is not None and t - last < every:
            continue
        img = decode(read())
        if img is None:
            continue
        last = t
        yield Frame(i, t, img, label)


def _stream(url: str, every: float) -> Iterator[Frame]:
    """Живой поток. Кадры читаются в отдельном потоке, наружу отдаётся самый свежий
    раз в `every` секунд — иначе при медленной обработке копилась бы задержка."""
    latest: dict = {}
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            cap = cv2.VideoCapture(url)
            while not stop.is_set() and cap.isOpened():
                ok, img = cap.read()
                if not ok:
                    break
                latest["img"] = img
            cap.release()
            if not stop.is_set():
                print(f"Поток {url} оборвался, переподключаюсь через 5 с")
                time.sleep(5)

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.monotonic()
    i = 0
    try:
        while True:
            img = latest.pop("img", None)
            if img is None:
                time.sleep(0.05)
                continue
            t = time.monotonic() - t0
            yield Frame(i, t, img, time.strftime("%Y-%m-%d %H:%M:%S"))
            i += 1
            time.sleep(max(0.0, every - (time.monotonic() - t0 - t)))
    finally:
        stop.set()


def is_stream(src: str) -> bool:
    return src.startswith(("http://", "https://", "rtsp://"))


def is_sequence(src: str) -> bool:
    p = Path(src)
    return not is_stream(src) and (p.is_dir() or p.suffix.lower() == ".zip")


def sequence_entries(src: str) -> list[tuple[str, callable]]:
    """Кадры из папки (рекурсивно) или zip: (имя, функция чтения байтов)."""
    path = Path(src)
    if path.is_dir():
        return [(str(p), p.read_bytes) for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXT]
    z = zipfile.ZipFile(path)
    return [(n, (lambda n=n: z.read(n))) for n in z.namelist() if Path(n).suffix.lower() in IMAGE_EXT]


def frames_from_entries(entries: list[tuple[str, callable]], every: float) -> Iterator[Frame]:
    return _sequence(entries, every)


def open_source(src: str, every: float) -> Iterator[Frame]:
    """src — путь к видео, папке или zip с кадрами, либо URL потока (http/rtsp)."""
    if is_stream(src):
        return _stream(src, every)
    if is_sequence(src):
        return _sequence(sequence_entries(src), every)
    if Path(src).suffix.lower() in VIDEO_EXT:
        return _video(Path(src), every)
    raise SystemExit(f"Не понимаю источник {src}: нужен видеофайл, папка или zip с кадрами, либо URL потока")
