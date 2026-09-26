"""Стоковая прошивка FlashForge AD5M / AD5M Pro.

HTTP 8898 — основной канал (нужны серийный номер и код доступа из меню «Сеть → Только LAN»):
    POST /detail   {serialNumber, checkCode}                → {"code":0, "detail":{...}}
    POST /control  {serialNumber, checkCode, payload:{cmd, args}}
        jobCtl_cmd       args {jobID:"", action: pause|continue|cancel}
        lightControl_cmd args {status: open|close}
    code 0 — успех; 1 — неверный код; -2 — не включён режим «Только LAN».
TCP 8899 — старый G-code протокол без авторизации, одна сессия на принтер:
    ~M601 S1 вход, ~M602 выход, ~M115 сведения (серийник), ~M119 статус, ~M27 прогресс,
    ~M105 температуры, ~M25 пауза, ~M24 продолжить, ~M26 стоп, ~M146 свет.
    Ответ: «CMD Mxxx Received.» … «ok».
Протокол сверен с ff-5mp-api-py (MIT) и Parallel-7/flashforge-api-docs.
"""
from __future__ import annotations

import re
import socket
import threading

import requests

from guard.domain.models import PrinterConfig, PrinterInfo, PrinterState, Protocol, Status
from guard.ports import AuthError, PrinterError

HTTP_STATUS = {
    "ready": Status.IDLE, "busy": Status.BUSY, "calibrate_doing": Status.BUSY, "downloading": Status.BUSY,
    "error": Status.ERROR, "heating": Status.HEATING, "printing": Status.PRINTING, "pausing": Status.PAUSING,
    "paused": Status.PAUSED, "pause": Status.PAUSED, "cancel": Status.CANCELLED, "completed": Status.COMPLETED,
}


def _num(v) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


class FlashForgeHttp:
    def __init__(self, cfg: PrinterConfig, timeout: float = 4.0):
        self.cfg = cfg
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def configured(self) -> bool:
        return bool(self.cfg.serial and self.cfg.check_code)

    def _post(self, path: str, payload: dict | None = None) -> dict:
        body = {"serialNumber": self.cfg.serial, "checkCode": self.cfg.check_code}
        if payload is not None:
            body["payload"] = payload
        url = f"http://{self.cfg.host}:{self.cfg.http_port}{path}"
        try:
            r = self.session.post(url, json=body, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            raise PrinterError(f"нет ответа по HTTP {self.cfg.http_port}: {type(e).__name__}") from e
        except ValueError as e:
            raise PrinterError("принтер ответил не JSON") from e
        code = data.get("code")
        if code == 0:
            return data
        msg = data.get("message") or f"код {code}"
        if code == 1:
            raise AuthError(f"неверный серийный номер или код доступа ({msg})")
        if code == -2:
            raise AuthError("на принтере не включён режим «Только LAN»")
        raise PrinterError(msg)

    def detail(self) -> dict:
        return dict(self._post("/detail").get("detail") or {})

    def state(self) -> PrinterState:
        d = self.detail()
        nozzle = d.get("rightTemp") if d.get("rightTemp") is not None else d.get("leftTemp")
        if nozzle is None and d.get("nozzleTemps"):
            nozzle = d["nozzleTemps"][0]
        nozzle_t = d.get("rightTargetTemp") if d.get("rightTargetTemp") is not None else d.get("leftTargetTemp")
        err = d.get("errorCode")
        return PrinterState(
            status=HTTP_STATUS.get(str(d.get("status", "")).lower(), Status.UNKNOWN),
            progress=_num(d.get("printProgress")), file=d.get("printFileName") or None,
            layer=d.get("printLayer"), total_layers=d.get("targetPrintLayer"),
            nozzle_temp=_num(nozzle), nozzle_target=_num(nozzle_t),
            bed_temp=_num(d.get("platTemp")), bed_target=_num(d.get("platTargetTemp")),
            elapsed_s=_num(d.get("printDuration")), remaining_s=_num(d.get("estimatedTime")),
            light=None if d.get("lightStatus") is None else d.get("lightStatus") == "open",
            message=None if err in (None, "", "0") else f"ошибка прошивки {err}")

    def job(self, action: str) -> None:
        self._post("/control", {"cmd": "jobCtl_cmd", "args": {"jobID": "", "action": action}})

    def light(self, on: bool) -> None:
        self._post("/control", {"cmd": "lightControl_cmd", "args": {"status": "open" if on else "close"}})


class FlashForgeTcp:
    """Одна постоянная сессия; при обрыве переподключается."""

    def __init__(self, cfg: PrinterConfig, timeout: float = 4.0):
        self.cfg = cfg
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    def _connect(self) -> None:
        try:
            s = socket.create_connection((self.cfg.host, self.cfg.tcp_port), timeout=self.timeout)
        except OSError as e:
            raise PrinterError(f"нет ответа по TCP {self.cfg.tcp_port}: {e.__class__.__name__}") from e
        s.settimeout(self.timeout)
        self._sock = s
        try:
            resp = self._exchange("~M601 S1")
        except PrinterError:
            resp = "Control failed."
        if "Control failed" in resp or "Control Success" not in resp:
            self._drop()
            raise PrinterError("принтер не пустил в TCP-сессию — её заняла другая программа (FlashPrint, Orca)")

    def _exchange(self, cmd: str) -> str:
        assert self._sock is not None
        self._sock.sendall((cmd + "\r\n").encode("ascii"))
        buf = b""
        while not buf.rstrip().endswith(b"ok"):          # ответ заканчивается строкой «ok»
            chunk = self._sock.recv(4096)
            if not chunk:
                raise PrinterError("принтер закрыл TCP-соединение")
            buf += chunk
            if len(buf) > 65536:
                raise PrinterError("слишком длинный ответ по TCP")
        return buf.decode("utf-8", errors="replace")          # имя принтера может быть не латиницей

    def _drop(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None

    def command(self, cmd: str) -> str:
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._sock is None:
                        self._connect()
                    return self._exchange(cmd)
                except (OSError, PrinterError) as e:
                    self._drop()
                    if attempt == 2:
                        raise e if isinstance(e, PrinterError) else PrinterError(f"TCP: {e}") from e
        raise PrinterError("TCP: не удалось")

    def info(self) -> dict[str, str]:
        out = {}
        for line in self.command("~M115").splitlines():
            if ":" in line and not line.startswith("CMD"):
                k, v = line.split(":", 1)
                out[k.strip().lower()] = v.strip()
        return out

    def state(self) -> PrinterState:
        st = self.command("~M119")
        machine = re.search(r"MachineStatus:\s*(\w+)", st)
        move = re.search(r"MoveMode:\s*(\w+)", st)
        led = re.search(r"LEDStatus:\s*(\d+|on|off)", st, re.I)
        cur = re.search(r"CurrentFile:\s*(.*)", st)
        m, mv = (machine.group(1) if machine else ""), (move.group(1) if move else "")
        if m == "BUILDING_FROM_SD":
            status = Status.PAUSED if mv == "PAUSED" else Status.PRINTING
        else:
            status = {"READY": Status.IDLE, "PAUSED": Status.PAUSED, "BUILDING_COMPLETED": Status.COMPLETED,
                      "BUSY": Status.BUSY}.get(m, Status.UNKNOWN)
        progress = layer = total = None
        pr = self.command("~M27")
        b = re.search(r"SD printing byte (\d+)/(\d+)", pr)
        if b and int(b.group(2)):
            progress = int(b.group(1)) / int(b.group(2))
        lay = re.search(r"Layer:\s*(\d+)/(\d+)", pr)
        if lay:
            layer, total = int(lay.group(1)), int(lay.group(2))
        t = self.command("~M105")
        tn = re.search(r"T0?:\s*([\d.]+)(?:/([\d.]+))?", t)
        tb = re.search(r"B:\s*([\d.]+)(?:/([\d.]+))?", t)
        return PrinterState(
            status=status, progress=progress, layer=layer, total_layers=total,
            file=(cur.group(1).strip() or None) if cur else None,
            nozzle_temp=_num(tn.group(1)) if tn else None, nozzle_target=_num(tn.group(2)) if tn else None,
            bed_temp=_num(tb.group(1)) if tb else None, bed_target=_num(tb.group(2)) if tb else None,
            light=(led.group(1).lower() in ("1", "on")) if led else None)

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.sendall(b"~M602\r\n")
                except OSError:
                    pass
            self._drop()


class FlashForgeDriver:
    """HTTP 8898, если заданы серийник и код доступа; иначе и при отказе HTTP — TCP 8899."""

    def __init__(self, cfg: PrinterConfig, timeout: float = 4.0):
        self.cfg = cfg
        self.http = FlashForgeHttp(cfg, timeout)
        self.tcp = FlashForgeTcp(cfg, timeout)

    def probe(self) -> PrinterInfo:
        info = self.tcp.info()
        serial = info.get("sn") or info.get("serial number", "")
        if self.cfg.check_code:
            if serial and not self.cfg.serial:
                self.cfg.serial = serial
            self.http.detail()          # проверит код доступа; AuthError уйдёт в панель как есть
        return PrinterInfo(Protocol.FLASHFORGE, model=info.get("machine type", ""), name=info.get("machine name", ""),
                           serial=serial, firmware=info.get("firmware", ""),
                           camera_url=f"http://{self.cfg.host}:8080/?action=snapshot")

    def state(self) -> PrinterState:
        if self.http.configured:
            try:
                return self.http.state()
            except AuthError:
                raise
            except PrinterError:
                pass
        return self.tcp.state()

    def _control(self, http_action: str, tcp_cmd: str) -> None:
        if self.http.configured:
            try:
                return self.http.job(http_action)
            except AuthError:
                raise
            except PrinterError:
                pass                     # сеть по HTTP легла — пробуем TCP
        resp = self.tcp.command(tcp_cmd)
        if "ok" not in resp:
            raise PrinterError(f"принтер не подтвердил {tcp_cmd}")

    def pause(self) -> None:
        self._control("pause", "~M25")

    def resume(self) -> None:
        self._control("continue", "~M24")

    def cancel(self) -> None:
        self._control("cancel", "~M26")

    def set_light(self, on: bool) -> None:
        if self.http.configured:
            return self.http.light(on)
        self.tcp.command("~M146 r255 g255 b255 F0" if on else "~M146 r0 g0 b0 F0")

    def close(self) -> None:
        self.tcp.close()
