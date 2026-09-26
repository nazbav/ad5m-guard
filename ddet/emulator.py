"""Эмулятор FlashForge AD5M: проигрывает записанную печать как живой принтер.

Камера — как mjpg-streamer у AD5M: GET /?action=snapshot и /?action=stream.
API — как у AD5M на порту 8898: POST /detail и /control (см. ddet/printer.py).
Пауза останавливает «печать» (кадры замирают), отмена завершает её.
"""
from __future__ import annotations

import bisect
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

import cv2

from ddet.frames import parse_timestamp
from ddet.sources import VIDEO_EXT, sequence_entries

BOUNDARY = "boundarydonotcross"  # как у mjpg-streamer


def load_session(src: str) -> list[tuple[float, Callable[[], bytes]]]:
    """Кадры записи: (секунды от начала, чтение JPEG). Видео, папка или zip с кадрами."""
    path = Path(src)
    if path.suffix.lower() in VIDEO_EXT:
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            frames.append((i / fps, (lambda b=jpg: b)))
            i += 1
        cap.release()
        return frames
    entries = sequence_entries(src)
    stamped = sorted(((parse_timestamp(Path(n).name), r) for n, r in entries), key=lambda x: x[0])
    if not stamped or stamped[0][0] is None:
        raise SystemExit(f"{src}: нет кадров printer_ГГГГММДД_ччммсс.jpg")
    t0 = stamped[0][0]
    return [((ts - t0).total_seconds(), r) for ts, r in stamped]


@dataclass
class Playback:
    """Положение «печати» во времени. clock подменяется в тестах."""
    frames: list[tuple[float, Callable[[], bytes]]]
    speed: float = 1.0
    clock: Callable[[], float] = time.monotonic
    status: str = "printing"
    commands: list[str] = field(default_factory=list)   # что приходило в /control — для тестов и логов

    def __post_init__(self):
        self._lock = threading.Lock()
        self._times = [t for t, _ in self.frames]
        self._pos = 0.0                      # секунды записи
        self._since = self.clock()

    def _position(self) -> float:
        if self.status != "printing":
            return self._pos
        pos = self._pos + (self.clock() - self._since) * self.speed
        if pos >= self._times[-1]:
            self._pos, self._since, self.status = self._times[-1], self.clock(), "completed"
            return self._pos
        return pos

    def frame_index(self) -> int:
        with self._lock:
            return max(0, bisect.bisect_right(self._times, self._position()) - 1)

    def snapshot(self) -> bytes:
        return self.frames[self.frame_index()][1]()

    def progress(self) -> float:
        with self._lock:
            return self._position() / max(self._times[-1], 1e-9)

    def control(self, action: str) -> bool:
        with self._lock:
            pos = self._position()
            self.commands.append(action)
            if action == "pause" and self.status == "printing":
                self._pos, self.status = pos, "paused"
            elif action == "continue" and self.status == "paused":
                self._since, self.status = self.clock(), "printing"
            elif action == "cancel" and self.status in ("printing", "paused"):
                self._pos, self.status = pos, "cancel"
            else:
                return False
            return True

    def current_status(self) -> str:
        with self._lock:
            self._position()
            return self.status


class EmulatedPrinter:
    """Камера и API одного эмулированного принтера на двух портах (0 — выбрать свободные)."""

    def __init__(self, playback: Playback, serial: str = "EMU0001", check_code: str = "0000",
                 host: str = "127.0.0.1", camera_port: int = 0, api_port: int = 0, name: str = "AD5M-emu",
                 stream_fps: float = 5.0):
        self.playback, self.serial, self.check_code, self.name = playback, serial, check_code, name
        emu = self

        class Camera(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                action = parse_qs(urlparse(self.path).query).get("action", ["stream"])[0]
                if action == "snapshot":
                    jpg = emu.playback.snapshot()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    return
                self.send_response(200)
                self.send_header("Content-Type", f"multipart/x-mixed-replace;boundary={BOUNDARY}")
                self.end_headers()
                try:
                    while True:
                        jpg = emu.playback.snapshot()
                        self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                         f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                        time.sleep(1 / stream_fps)
                except (ConnectionError, OSError):
                    pass

        class Api(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, data: dict):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                except ValueError:
                    return self._reply({"code": 1, "message": "Bad JSON"})
                if req.get("serialNumber") != emu.serial or req.get("checkCode") != emu.check_code:
                    return self._reply({"code": 1, "message": "Unauthorized"})
                path = urlparse(self.path).path
                if path == "/detail":
                    return self._reply({"code": 0, "message": "Success", "detail": {
                        "name": emu.name, "status": emu.playback.current_status(),
                        "printProgress": round(emu.playback.progress(), 4), "printFileName": "emulated.gcode"}})
                if path == "/control":
                    payload = req.get("payload") or {}
                    if payload.get("cmd") != "jobCtl_cmd":
                        return self._reply({"code": 1, "message": "Unsupported cmd"})
                    ok = emu.playback.control((payload.get("args") or {}).get("action", ""))
                    return self._reply({"code": 0 if ok else 1, "message": "Success" if ok else "Invalid state"})
                self._reply({"code": 1, "message": "Unknown path"})

        self.camera_server = ThreadingHTTPServer((host, camera_port), Camera)
        self.api_server = ThreadingHTTPServer((host, api_port), Api)
        self.camera_server.daemon_threads = self.api_server.daemon_threads = True
        self.host = host

    @property
    def camera_port(self) -> int:
        return self.camera_server.server_address[1]

    @property
    def api_port(self) -> int:
        return self.api_server.server_address[1]

    def start(self) -> "EmulatedPrinter":
        for s in (self.camera_server, self.api_server):
            threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def stop(self) -> None:
        for s in (self.camera_server, self.api_server):
            s.shutdown()
            s.server_close()
