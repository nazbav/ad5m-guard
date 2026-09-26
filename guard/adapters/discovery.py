"""Поиск принтеров в локальной сети.

Проверяются адреса /24-подсетей сетевых карт этого компьютера (только частные адреса) на портах
8898 / 8899 (стоковая прошивка FlashForge) и 7125 (Moonraker). Кто ответил — опознаётся так же, как
«Проверить подключение»: FlashForge по TCP ~M115 без кода отдаёт модель, имя и серийный номер,
Moonraker — /printer/info. Всё параллельно, подсеть /24 — несколько секунд.
"""
from __future__ import annotations

import ipaddress
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from guard.adapters.drivers import probe
from guard.domain.models import PrinterConfig
from guard.ports import AuthError, PrinterError

FF_HTTP, FF_TCP, MOONRAKER = 8898, 8899, 7125
MAX_HOSTS = 1024


def local_subnets(limit: int = 4) -> list[ipaddress.IPv4Network]:
    """/24 вокруг каждого частного IPv4 этого компьютера (без loopback и 169.254.*)."""
    ips: list[str] = []
    try:
        ips += [i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    try:                                    # адрес карты, через которую идёт маршрут по умолчанию
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.insert(0, s.getsockname()[0])
    except OSError:
        pass
    nets: list[ipaddress.IPv4Network] = []
    for ip in ips:
        a = ipaddress.IPv4Address(ip)
        if a.is_private and not a.is_loopback and not a.is_link_local:
            n = ipaddress.IPv4Network(f"{ip}/24", strict=False)
            if n not in nets:
                nets.append(n)
    return nets[:limit]


# Адреса самого компьютера: эмулятор принтеров (python -m emulator) и принтеры, проброшенные сюда.
# 14 адресов — проверка мгновенная; в обычной работе там никого нет.
THIS_PC = ipaddress.IPv4Network("127.0.0.0/28")


def default_subnets() -> list[ipaddress.IPv4Network]:
    return local_subnets() + [THIS_PC]


def parse_subnet(text: str) -> ipaddress.IPv4Network:
    try:
        net = ipaddress.IPv4Network(text.strip(), strict=False)
    except ValueError as e:
        raise ValueError(f"подсеть вида 192.168.1.0/24: {e}") from e
    if net.num_addresses > MAX_HOSTS + 2:
        raise ValueError(f"слишком большая подсеть — не больше {MAX_HOSTS} адресов (например /24)")
    return net


def _open(host: str, port: int, timeout: float) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def identify(host: str, ports: set[int]) -> dict | None:
    """Кто на этом адресе. None — не принтер."""
    kind = "moonraker" if MOONRAKER in ports and not ports & {FF_HTTP, FF_TCP} else "flashforge"
    cfg = PrinterConfig(name="поиск", host=host, protocol=kind)
    found = {"host": host, "protocol": kind, "model": "", "name": "", "serial": "", "camera_url": "", "note": ""}
    try:
        info = probe(cfg, timeout=3.0)
        found.update(protocol=info.protocol.value, model=info.model, name=info.name,
                     serial=info.serial or cfg.serial, camera_url=info.camera_url)
    except AuthError:
        found["note"] = "нужен код доступа"
    except PrinterError as e:
        if kind == "flashforge" and {FF_HTTP, FF_TCP} <= ports:
            found["note"] = f"похоже на FlashForge, но не ответил: {e}"   # TCP занят FlashPrint/Orca и т.п.
        else:
            return None
    return found


def scan(hosts: list[str], progress: Callable[[int, int], None] | None = None,
         cancel: threading.Event | None = None, timeout: float = 0.5, ports=(FF_HTTP, FF_TCP, MOONRAKER)) -> list[dict]:
    """Порты всех адресов параллельно, затем опознание тех, где что-то открыто."""
    total, done = len(hosts), 0
    lock = threading.Lock()
    open_ports: dict[str, set[int]] = {}

    def check(host: str) -> None:
        nonlocal done
        if not (cancel and cancel.is_set()):
            got = {p for p in ports if _open(host, p, timeout)}
            if got:
                with lock:
                    open_ports[host] = got
        with lock:
            done += 1
            if progress:
                progress(done, total)

    with ThreadPoolExecutor(min(128, max(1, total))) as pool:
        list(pool.map(check, hosts))
    if cancel and cancel.is_set():
        return []
    with ThreadPoolExecutor(min(16, max(1, len(open_ports)))) as pool:
        found = list(pool.map(lambda h: identify(h, open_ports[h]), sorted(open_ports, key=ipaddress.IPv4Address)))
    return [f for f in found if f]
