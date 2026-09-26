"""Камера по HTTP: снимок (mjpg-streamer ?action=snapshot) или первый кадр из MJPEG-потока (?action=stream)."""
from __future__ import annotations

import requests

from guard.domain.models import PrinterConfig
from guard.ports import PrinterError

SOI, EOI = b"\xff\xd8", b"\xff\xd9"
MAX_FRAME = 8 * 1024 * 1024


class HttpCamera:
    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()

    @classmethod
    def for_printer(cls, cfg: PrinterConfig) -> "HttpCamera":
        return cls(cfg.default_camera_url())

    def snapshot(self) -> bytes:
        try:
            with self.session.get(self.url, timeout=self.timeout, stream=True) as r:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                if "multipart" in ctype:
                    return self._first_frame(r)
                data = r.raw.read(MAX_FRAME + 1, decode_content=True)
        except requests.RequestException as e:
            raise PrinterError(f"камера не отвечает ({type(e).__name__})") from e
        if len(data) > MAX_FRAME or not data.startswith(SOI):
            raise PrinterError("камера отдала не JPEG")
        return data

    def _first_frame(self, r) -> bytes:
        buf = b""
        for chunk in r.iter_content(16384):
            buf += chunk
            start = buf.find(SOI)
            end = buf.find(EOI, start + 2) if start >= 0 else -1
            if end >= 0:
                return buf[start:end + 2]
            if len(buf) > MAX_FRAME:
                break
        raise PrinterError("в потоке камеры нет кадра")

    def close(self) -> None:
        self.session.close()
