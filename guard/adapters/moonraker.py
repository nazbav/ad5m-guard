"""Прошивки на Klipper (ZMOD, Klipper Mod для AD5M): Moonraker API, порт 7125.

    GET  /printer/objects/query?print_stats&virtual_sdcard&extruder&heater_bed&webhooks
    POST /printer/print/pause | resume | cancel
    GET  /server/info, /printer/info, /server/webcams/list   (проверка подключения, адрес камеры)
Если на Moonraker включена авторизация, ключ API задаётся в поле «код доступа» (заголовок X-Api-Key).
"""
from __future__ import annotations

import requests

from guard.domain.models import PrinterConfig, PrinterInfo, PrinterState, Protocol, Status
from guard.ports import AuthError, PrinterError

STATES = {"standby": Status.IDLE, "printing": Status.PRINTING, "paused": Status.PAUSED,
          "complete": Status.COMPLETED, "cancelled": Status.CANCELLED, "error": Status.ERROR}
QUERY = ("print_stats=state,filename,print_duration,message,info&virtual_sdcard=progress"
         "&extruder=temperature,target&heater_bed=temperature,target&webhooks=state,state_message"
         "&display_status=progress")


class MoonrakerDriver:
    def __init__(self, cfg: PrinterConfig, timeout: float = 4.0):
        self.cfg = cfg
        self.timeout = timeout
        self.base = f"http://{cfg.host}:{cfg.moonraker_port}"
        self.session = requests.Session()
        if cfg.check_code:
            self.session.headers["X-Api-Key"] = cfg.check_code

    def _req(self, method: str, path: str) -> dict:
        try:
            r = self.session.request(method, self.base + path, timeout=self.timeout)
        except requests.RequestException as e:
            raise PrinterError(f"нет ответа Moonraker {self.cfg.moonraker_port}: {type(e).__name__}") from e
        if r.status_code in (401, 403):
            raise AuthError("Moonraker требует ключ API (поле «код доступа»)")
        if r.status_code >= 400:
            try:
                msg = r.json().get("error", {}).get("message", r.text[:200])
            except ValueError:
                msg = r.text[:200]
            raise PrinterError(f"Moonraker {r.status_code}: {msg}")
        try:
            return r.json().get("result", {})
        except ValueError as e:
            raise PrinterError("Moonraker ответил не JSON") from e

    def probe(self) -> PrinterInfo:
        server = self._req("GET", "/server/info")
        info = self._req("GET", "/printer/info")
        cam = ""
        try:
            cams = self._req("GET", "/server/webcams/list").get("webcams", [])
            if cams:
                url = cams[0].get("snapshot_url") or cams[0].get("stream_url") or ""
                cam = url if url.startswith("http") else f"http://{self.cfg.host}{url}"
        except PrinterError:
            pass
        return PrinterInfo(Protocol.MOONRAKER, model="Klipper", name=info.get("hostname", ""),
                           firmware=f"Klipper {info.get('software_version', '')}; Moonraker {server.get('moonraker_version', '')}",
                           camera_url=cam or f"http://{self.cfg.host}:8080/?action=snapshot")

    def state(self) -> PrinterState:
        s = self._req("GET", f"/printer/objects/query?{QUERY}").get("status", {})
        ps, wh = s.get("print_stats", {}), s.get("webhooks", {})
        status = STATES.get(ps.get("state", ""), Status.UNKNOWN)
        message = ps.get("message") or None
        if wh.get("state") in ("shutdown", "error"):
            status, message = Status.ERROR, wh.get("state_message") or message
        info = ps.get("info") or {}
        progress = s.get("virtual_sdcard", {}).get("progress")
        if progress is None:
            progress = s.get("display_status", {}).get("progress")
        elapsed = ps.get("print_duration")
        remaining = elapsed / progress - elapsed if elapsed and progress and progress > 0.01 else None
        ext, bed = s.get("extruder", {}), s.get("heater_bed", {})
        return PrinterState(
            status=status, progress=progress, file=ps.get("filename") or None,
            layer=info.get("current_layer"), total_layers=info.get("total_layer"),
            nozzle_temp=ext.get("temperature"), nozzle_target=ext.get("target"),
            bed_temp=bed.get("temperature"), bed_target=bed.get("target"),
            elapsed_s=elapsed, remaining_s=remaining, message=message)

    def pause(self) -> None:
        self._req("POST", "/printer/print/pause")

    def resume(self) -> None:
        self._req("POST", "/printer/print/resume")

    def cancel(self) -> None:
        self._req("POST", "/printer/print/cancel")

    def set_light(self, on: bool) -> None:
        raise PrinterError("управление светом через Moonraker не поддерживается")

    def close(self) -> None:
        self.session.close()
