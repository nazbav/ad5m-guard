"""Сквозной тест: настоящие драйверы и камера по сети → эмулятор AD5M, сервис парка, хранилища на диске."""
import itertools

import pytest

from emulator.servers import PrinterNode
from emulator.virtual import VirtualPrinter, synthetic_frames
from guard.adapters.camera import HttpCamera
from guard.adapters.drivers import make_driver
from guard.adapters.storage import SqliteEventStore, YamlConfigStore
from guard.domain.models import Detection, PrinterConfig
from guard.domain.policy import Settings
from guard.services.fleet import FleetService
from guard.web.app import create_app

_hosts = itertools.count(60)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class BlobDetector:
    """Светлое пятно в центре синтетического кадра — «спагетти»."""
    has_scene = False

    def detect(self, images):
        return [[Detection("defects", "spaghetti", 0.9, (260, 280, 380, 400))] if im[330:350, 310:330].mean() > 200 else []
                for im in images]


@pytest.fixture
def fleet(tmp_path):
    clock = Clock()
    nodes, cfgs = [], []
    for kind, bad in (("flashforge", 20), ("flashforge", None), ("moonraker", 20)):
        p = VirtualPrinter(f"emu-{kind}-{bad}", "SN1", "code", kind=kind, heat_s=0, clock=clock)
        p.start(synthetic_frames(80, 3.0, bad_from=bad))
        n = PrinterNode(p, f"127.0.0.{next(_hosts)}", {"http": 0, "tcp": 0, "moonraker": 0, "camera": 0}).start()
        nodes.append(n)
        extra = ({"protocol": "moonraker", "moonraker_port": n.port("moonraker")} if kind == "moonraker" else
                 {"protocol": "flashforge", "http_port": n.port("http"), "tcp_port": n.port("tcp"),
                  "serial": "SN1", "check_code": "code"})
        cfgs.append(PrinterConfig(name=p.name, host=n.host, camera_url=f"http://{n.host}:{n.port('camera')}/?action=snapshot", **extra))
    store = YamlConfigStore(tmp_path / "config.yaml")
    store.save(Settings(window=3, min_hits=2), cfgs)
    events = SqliteEventStore(tmp_path / "events.db", tmp_path / "alerts")
    svc = FleetService(store, events, BlobDetector(), driver_factory=lambda c: make_driver(c, 2.0),
                       camera_factory=HttpCamera.for_printer, clock=clock)
    yield svc, nodes, clock, tmp_path
    svc.close()
    events.close()
    for n in nodes:
        n.stop()


def test_fleet_stops_only_failed_prints_over_network(fleet):
    svc, nodes, clock, tmp = fleet
    for k in range(40):                     # 40 циклов по 3 с «времени печати»
        clock.t = k * 3.0
        svc.step()
    bad_ff, good, bad_moon = (n.printer for n in nodes)
    assert bad_ff.commands == ["pause"] and bad_ff.status == "paused"
    assert bad_moon.commands == ["pause"] and bad_moon.status == "paused"
    assert good.commands == [] and good.status == "printing"
    ev = svc.events.recent(100)
    stopped = [e for e in ev if e.kind == "stopped"]
    assert len(stopped) == 2 and all(e.image and (tmp / "alerts" / e.image).exists() for e in stopped)


def test_restart_keeps_printers_and_history(fleet):
    svc, nodes, clock, tmp = fleet
    svc.step()
    svc.add_printer({"name": "added", "host": "192.168.1.77", "protocol": "moonraker"})
    again = FleetService(YamlConfigStore(tmp / "config.yaml"), svc.events, None, make_driver, HttpCamera.for_printer)
    assert {w.cfg.name for w in again.workers.values()} >= {"added", nodes[0].printer.name}
    assert any(e.detail.startswith("добавлен") for e in again.events.recent(50))
    again.close()


def test_web_manual_control_reaches_emulator(fleet):
    svc, nodes, clock, _ = fleet
    svc.step()
    c = create_app(svc).test_client()
    pid = next(w.cfg.id for w in svc.workers.values() if w.cfg.name == nodes[1].printer.name)
    assert c.post(f"/api/printers/{pid}/control/pause").status_code == 200
    assert nodes[1].printer.status == "paused"
    assert c.post(f"/api/printers/{pid}/control/resume").status_code == 200
    assert nodes[1].printer.status == "printing"
    r = c.post("/api/printers/test", json={"name": "t", "host": nodes[0].host, "protocol": "flashforge",
                                          "tcp_port": nodes[0].port("tcp"), "http_port": nodes[0].port("http")})
    assert r.status_code == 200 and r.get_json()["serial"] == "SN1"
