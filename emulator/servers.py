"""Сетевые интерфейсы виртуального принтера — те же порты и форматы, что у настоящего AD5M."""
from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from emulator.virtual import VirtualPrinter

BOUNDARY = "boundarydonotcross"   # как у mjpg-streamer
MOONRAKER_STATE = {"ready": "standby", "heating": "printing", "printing": "printing", "pausing": "paused",
                   "paused": "paused", "completed": "complete", "cancel": "cancelled", "error": "error"}


class _Http(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def server_bind(self):
        # HTTPServer делает socket.getfqdn(адрес) — на Windows для 127.0.0.x это ~10 с обратного DNS
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


class _Base(BaseHTTPRequestHandler):
    printer: VirtualPrinter

    def log_message(self, *a):
        pass

    def _offline(self) -> bool:
        if self.printer.faults.offline:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return True
        self.printer.delay()
        return False

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        try:
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            return {}


# --------------------------------------------------------------------- HTTP 8898 (стоковая прошивка)
class FlashForgeHttpHandler(_Base):
    def do_POST(self):
        if self._offline():
            return
        p, req = self.printer, self._body()
        if not p.faults.lan_only:
            return self._json({"code": -2, "message": "Please turn on the LAN mode"})
        if req.get("serialNumber") != p.serial or req.get("checkCode") != p.check_code:
            return self._json({"code": 1, "message": "Access code is different"})
        path = urlparse(self.path).path
        if path == "/detail":
            s = p.snapshot_state()
            status = "error" if s["error"] else s["status"]
            return self._json({"code": 0, "message": "Success", "detail": {
                "name": p.name, "model": p.model, "firmwareVersion": p.firmware, "macAddr": p.mac, "ipAddr": self.server.server_address[0],
                "status": status, "printFileName": s["file"], "printProgress": s["progress"],
                "printLayer": s["layer"], "targetPrintLayer": s["total_layers"],
                "rightTemp": s["nozzle"], "rightTargetTemp": s["nozzle_target"],
                "platTemp": s["bed"], "platTargetTemp": s["bed_target"],
                "printDuration": int(s["elapsed"]), "estimatedTime": int(s["remaining"]),
                "lightStatus": "open" if s["light"] else "close", "errorCode": s["error"] or "",
                "cameraStreamUrl": f"http://{self.server.server_address[0]}:8080/?action=stream"}})
        if path == "/product":
            return self._json({"code": 0, "message": "Success", "product": {
                "lightCtrlState": 1, "nozzleTempCtrlState": 1, "platformTempCtrlState": 1,
                "chamberTempCtrlState": 0, "internalFanCtrlState": 1, "externalFanCtrlState": 1}})
        if path == "/control":
            payload = req.get("payload") or {}
            cmd, args = payload.get("cmd"), payload.get("args") or {}
            if cmd == "jobCtl_cmd":
                action = {"pause": "pause", "continue": "resume", "cancel": "cancel"}.get(args.get("action"))
                ok = action is not None and p.control(action)
                return self._json({"code": 0 if ok else 1, "message": "Success" if ok else "Invalid state"})
            if cmd == "lightControl_cmd":
                p.set_light(args.get("status") == "open")
                return self._json({"code": 0, "message": "Success"})
            return self._json({"code": 1, "message": "Unsupported cmd"})
        self._json({"code": 1, "message": "Unknown path"}, 404)


# --------------------------------------------------------------------- TCP 8899 (G-code протокол)
class FlashForgeTcpHandler(socketserver.StreamRequestHandler):
    printer: VirtualPrinter
    server: "_Tcp"

    def _reply(self, cmd: str, body: str = "") -> None:
        text = f"CMD {cmd} Received.\r\n" + (body + "\r\n" if body else "") + "ok\r\n"
        self.wfile.write(text.encode("utf-8"))

    def handle(self):
        p = self.printer
        logged_in = False
        try:
            while True:
                line = self.rfile.readline()
                if not line or p.faults.offline:
                    return
                p.delay()
                raw = line.decode("ascii", errors="replace").strip().lstrip("~")
                if not raw:
                    continue
                code = raw.split()[0].upper()
                if code == "M601":
                    with self.server.session_lock:
                        if self.server.session_active:
                            self.wfile.write(b"CMD M601 Received.\r\nControl failed.\r\nok\r\n")
                            return
                        self.server.session_active = logged_in = True
                    self._reply("M601", "Control Success V2.1.")
                    continue
                if not logged_in:
                    self.wfile.write(f"CMD {code} Received.\r\nPlease login first.\r\nok\r\n".encode())
                    continue
                if code == "M602":
                    self._reply("M602", "Control Release.")
                    return
                self._command(code, raw)
        finally:
            if logged_in:
                with self.server.session_lock:
                    self.server.session_active = False

    def _command(self, code: str, raw: str) -> None:
        p = self.printer
        s = p.snapshot_state()
        if code == "M115":
            self._reply(code, f"Machine Type: Flashforge {p.model}\r\nMachine Name: {p.name}\r\nFirmware: v{p.firmware}\r\n"
                              f"SN: {p.serial}\r\nX: 220 Y: 220 Z: 220\r\nTool Count: 1\r\nMac Address:{p.mac}")
        elif code == "M119":
            st = s["status"]
            machine = {"ready": "READY", "cancel": "READY", "completed": "BUILDING_COMPLETED"}.get(st, "BUILDING_FROM_SD")
            move = "PAUSED" if st in ("paused", "pausing") else ("READY" if machine != "BUILDING_FROM_SD" else "MOVING")
            self._reply(code, f"Endstop: X-max:0 Y-max:0 Z-min:0\r\nMachineStatus: {machine}\r\nMoveMode: {move}\r\n"
                              f"Status: S:1 L:0 J:0 F:0\r\nLED: {1 if s['light'] else 0}\r\nCurrentFile: {s['file']}")
        elif code == "M27":
            total = 1_000_000
            self._reply(code, f"SD printing byte {int(s['progress'] * total)}/{total}\r\nLayer: {s['layer']}/{s['total_layers']}")
        elif code == "M105":
            self._reply(code, f"T0:{s['nozzle']:.0f}/{s['nozzle_target']:.0f} T1:0/0 B:{s['bed']:.0f}/{s['bed_target']:.0f}")
        elif code in ("M25", "M24", "M26"):
            ok = p.control({"M25": "pause", "M24": "resume", "M26": "cancel"}[code])
            self._reply(code, "" if ok else "Error: invalid state")
        elif code == "M146":
            p.set_light(" r0 " not in f" {raw} ")
            self._reply(code)
        else:
            self._reply(code)


class _Tcp(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.session_lock = threading.Lock()
        self.session_active = False


# --------------------------------------------------------------------- Moonraker 7125 (Klipper)
class MoonrakerHandler(_Base):
    def _ok(self, result) -> None:
        self._json({"result": result})

    def do_GET(self):
        if self._offline():
            return
        p, u = self.printer, urlparse(self.path)
        if u.path == "/server/info":
            return self._ok({"klippy_connected": True, "klippy_state": "ready", "moonraker_version": "v0.9.3-emu"})
        if u.path == "/printer/info":
            return self._ok({"state": "ready", "hostname": p.name, "software_version": "v0.12.0-297-g2c3b0e5c"})
        if u.path == "/server/webcams/list":
            host = self.server.server_address[0]
            return self._ok({"webcams": [{"name": "cam", "snapshot_url": f"http://{host}:8080/?action=snapshot",
                                          "stream_url": f"http://{host}:8080/?action=stream"}]})
        if u.path == "/printer/objects/query":
            s = p.snapshot_state()
            state = "error" if s["error"] else MOONRAKER_STATE.get(s["status"], "standby")
            return self._ok({"eventtime": time.time(), "status": {
                "print_stats": {"state": state, "filename": s["file"], "print_duration": s["elapsed"],
                                "message": s["error"], "info": {"current_layer": s["layer"], "total_layer": s["total_layers"]}},
                "virtual_sdcard": {"progress": s["progress"]},
                "display_status": {"progress": s["progress"]},
                "extruder": {"temperature": s["nozzle"], "target": s["nozzle_target"]},
                "heater_bed": {"temperature": s["bed"], "target": s["bed_target"]},
                "webhooks": {"state": "ready", "state_message": "Printer is ready"}}})
        self._json({"error": {"code": 404, "message": "Not Found"}}, 404)

    def do_POST(self):
        if self._offline():
            return
        action = {"/printer/print/pause": "pause", "/printer/print/resume": "resume",
                  "/printer/print/cancel": "cancel"}.get(urlparse(self.path).path)
        if action is None:
            return self._json({"error": {"code": 404, "message": "Not Found"}}, 404)
        if not self.printer.control(action):
            return self._json({"error": {"code": 400, "message": f"Cannot {action} in current state"}}, 400)
        self._ok("ok")


# --------------------------------------------------------------------- камера 8080 (mjpg-streamer)
class CameraHandler(_Base):
    fps = 5.0

    def do_GET(self):
        if self._offline() or self.printer.faults.camera_down:
            if self.printer.faults.camera_down:
                self.send_error(503, "camera not available")
            return
        action = parse_qs(urlparse(self.path).query).get("action", ["stream"])[0]
        if action == "snapshot":
            jpg = self.printer.frame()
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
            while not self.printer.faults.camera_down and not self.printer.faults.offline:
                jpg = self.printer.frame()
                self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n".encode()
                                 + jpg + b"\r\n")
                time.sleep(1 / self.fps)
        except (ConnectionError, OSError):
            pass


def _bind(handler_base, printer):
    return type(handler_base.__name__, (handler_base,), {"printer": printer})


class PrinterNode:
    """Все сетевые интерфейсы одного виртуального принтера на одном адресе."""

    def __init__(self, printer: VirtualPrinter, host: str = "127.0.0.2", ports: dict[str, int] | None = None):
        ports = {"http": 8898, "tcp": 8899, "moonraker": 7125, "camera": 8080} | (ports or {})
        self.printer, self.host = printer, host
        self.servers = [_Http((host, ports["camera"]), _bind(CameraHandler, printer))]
        if printer.kind == "moonraker":
            self.servers.append(_Http((host, ports["moonraker"]), _bind(MoonrakerHandler, printer)))
        else:
            self.servers.append(_Http((host, ports["http"]), _bind(FlashForgeHttpHandler, printer)))
            self.servers.append(_Tcp((host, ports["tcp"]), _bind(FlashForgeTcpHandler, printer)))

    def port(self, which: str) -> int:
        idx = {"camera": 0}.get(which)
        if idx is None:
            wanted = {"moonraker": MoonrakerHandler, "http": FlashForgeHttpHandler, "tcp": FlashForgeTcpHandler}[which]
            idx = next(i for i, s in enumerate(self.servers) if issubclass(s.RequestHandlerClass, wanted))
        return self.servers[idx].server_address[1]

    def start(self) -> "PrinterNode":
        for s in self.servers:
            threading.Thread(target=s.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        return self

    def stop(self) -> None:
        for s in self.servers:
            s.shutdown()
            s.server_close()
