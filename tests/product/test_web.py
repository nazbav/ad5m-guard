import pytest

from fakes import FakeCamera, FakeDetector, FakeDriver, FakePrinter, MemConfig, MemEvents
from guard.domain.models import PrinterConfig
from guard.domain.policy import Settings
from guard.services.fleet import FleetService
from guard.web.app import create_app


@pytest.fixture
def ctx():
    cfg = PrinterConfig(name="p1", host="10.0.0.1", protocol="flashforge", serial="S", check_code="secret")
    printers = {cfg.id: FakePrinter()}
    fleet = FleetService(MemConfig(Settings(window=3, min_hits=2), [cfg]), MemEvents(), FakeDetector(),
                         driver_factory=lambda c: FakeDriver(printers.setdefault(c.id, FakePrinter())),
                         camera_factory=lambda c: FakeCamera(printers.setdefault(c.id, FakePrinter())))
    fleet.step(now=0)
    return fleet, printers, cfg


def test_state_hides_secrets_and_serves_frame(ctx):
    fleet, _, cfg = ctx
    c = create_app(fleet).test_client()
    s = c.get("/api/state").get_json()
    assert s["printers"][0]["check_code"] == "••••" and s["printers"][0]["state"]["status"] == "printing"
    assert "secret" not in c.get("/api/state").get_data(as_text=True)
    r = c.get(f"/api/frame/{cfg.id}.jpg")
    assert r.status_code == 200 and r.data[:2] == b"\xff\xd8"
    assert c.get("/api/frame/nope.jpg").status_code == 404
    assert "AD5M Guard" in c.get("/").get_data(as_text=True)


def test_printer_crud_and_validation(ctx):
    fleet, _, _ = ctx
    c = create_app(fleet).test_client()
    r = c.post("/api/printers", json={"name": "p2", "host": "10.0.0.2", "protocol": "moonraker"})
    assert r.status_code == 201
    pid = r.get_json()["id"]
    assert c.post("/api/printers", json={"name": "p2", "host": "10.0.0.3"}).status_code == 400
    assert c.post("/api/printers", json={"name": "x", "host": "bad host"}).status_code == 400
    assert c.put(f"/api/printers/{pid}", json={"ai_enabled": False}).get_json()["ai_enabled"] is False
    assert c.delete(f"/api/printers/{pid}").status_code == 200
    assert c.delete(f"/api/printers/{pid}").status_code == 404


def test_control_and_settings(ctx):
    fleet, printers, cfg = ctx
    c = create_app(fleet).test_client()
    assert c.post(f"/api/printers/{cfg.id}/control/pause").status_code == 200
    assert printers[cfg.id].commands == ["pause"]
    assert c.post(f"/api/printers/{cfg.id}/control/explode").status_code == 400
    r = c.put("/api/settings", json={"defect_conf": 0.8, "stop_on": ["spaghetti"]})
    assert r.status_code == 200 and fleet.settings.defect_conf == 0.8
    assert c.put("/api/settings", json={"min_hits": 9, "window": 3}).status_code == 400
    d = c.get("/api/settings/defaults").get_json()
    assert d["defect_conf"] == Settings().defect_conf and "telegram_token" not in d and "interval_s" not in d


def test_token_protects_writes_not_reads(ctx):
    fleet, printers, cfg = ctx
    c = create_app(fleet, token="t0k").test_client()
    assert c.get("/api/state").status_code == 200
    assert c.post(f"/api/printers/{cfg.id}/control/pause").status_code == 403
    assert c.put("/api/settings", json={"action": "none"}).status_code == 403
    assert printers[cfg.id].commands == []
    assert c.post(f"/api/printers/{cfg.id}/control/pause", headers={"X-Token": "t0k"}).status_code == 200


def test_printer_errors_mapped(ctx):
    fleet, printers, cfg = ctx
    printers[cfg.id].offline = True

    def boom():
        from guard.ports import PrinterError
        raise PrinterError("нет ответа")
    fleet.workers[cfg.id].driver = FakeDriver(printers[cfg.id])
    fleet.workers[cfg.id].driver.pause = boom
    c = create_app(fleet).test_client()
    r = c.post(f"/api/printers/{cfg.id}/control/pause")
    assert r.status_code == 502 and "нет ответа" in r.get_json()["error"]
