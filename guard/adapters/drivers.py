"""Выбор драйвера по протоколу и автоопределение протокола при добавлении принтера."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from guard.adapters.flashforge import FlashForgeDriver
from guard.adapters.moonraker import MoonrakerDriver
from guard.domain.models import PrinterConfig, PrinterInfo, Protocol
from guard.ports import AuthError, PrinterDriver, PrinterError


def make_driver(cfg: PrinterConfig, timeout: float = 4.0) -> PrinterDriver:
    if cfg.protocol is Protocol.MOONRAKER:
        return MoonrakerDriver(cfg, timeout)
    if cfg.protocol is Protocol.FLASHFORGE:
        return FlashForgeDriver(cfg, timeout)
    raise PrinterError("протокол не определён — нажмите «Проверить подключение»")


def probe(cfg: PrinterConfig, timeout: float = 3.0) -> PrinterInfo:
    """Проверка подключения. Для «авто» пробует оба протокола сразу; ошибка авторизации важнее «не отвечает»."""
    kinds = [cfg.protocol] if cfg.protocol is not Protocol.AUTO else [Protocol.FLASHFORGE, Protocol.MOONRAKER]

    def attempt(kind: Protocol):
        c = PrinterConfig.from_dict({**cfg.to_dict(), "protocol": kind.value})
        d = make_driver(c, timeout)
        try:
            info = d.probe()
            if c.serial and not cfg.serial:
                cfg.serial = c.serial
            return info
        finally:
            d.close()

    errors: list[PrinterError] = []
    with ThreadPoolExecutor(len(kinds)) as pool:
        for fut in [pool.submit(attempt, k) for k in kinds]:
            try:
                return fut.result()
            except PrinterError as e:
                errors.append(e)
    auth = next((e for e in errors if isinstance(e, AuthError)), None)
    if auth:
        raise auth
    raise PrinterError("принтер не отвечает: " + "; ".join(str(e) for e in errors))
