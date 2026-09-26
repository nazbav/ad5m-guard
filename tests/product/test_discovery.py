"""Поиск в сети против эмулятора: стоковый AD5M и Klipper на своих адресах, пустые адреса рядом."""
import time

from emulator.servers import PrinterNode
from emulator.virtual import VirtualPrinter
from guard.services.discovery import DiscoveryService


def test_finds_emulated_printers_and_hides_added():
    ff = VirtualPrinter("Цех-1", "SNFIND0001", "12345678", kind="flashforge")
    kl = VirtualPrinter("klipper-box", "", "", kind="moonraker")
    nodes = [PrinterNode(ff, "127.0.0.41").start(), PrinterNode(kl, "127.0.0.42").start()]
    try:
        known = {"127.0.0.42"}
        svc = DiscoveryService(lambda: known)
        svc.start("127.0.0.40/29")                          # .41–.46: два принтера и пустые адреса
        deadline = time.time() + 30
        while svc.status()["running"] and time.time() < deadline:
            time.sleep(0.2)
        s = svc.status()
        assert not s["running"] and s["done"] == s["total"] == 6 and s["error"] is None
        assert [(f["host"], f["protocol"], f["serial"]) for f in s["found"]] == [("127.0.0.41", "flashforge", "SNFIND0001")]
        assert s["already_added"] == 1                      # Klipper уже в парке — не предлагаем
    finally:
        for n in nodes:
            n.stop()
