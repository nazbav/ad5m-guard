import cv2
import numpy as np
import pytest

from ddet.dashboard import create_app
from ddet.detect import Detection
from ddet.emulator import EmulatedPrinter, Playback
from ddet.printer import PrinterSpec
from ddet.service import Monitor, ServiceConfig
from ddet.watch import WatchConfig


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class FakeDetector:
    models = {"defects": None}

    def batch(self, images):
        return [[Detection("defects", "spaghetti", 0.9, (10, 10, 50, 50))] if img.mean() > 128 else [] for img in images]


def session(bright_from=5, n=20):
    out = []
    for i in range(n):
        jpg = cv2.imencode(".jpg", np.full((480, 640, 3), 220 if i >= bright_from else 30, np.uint8))[1].tobytes()
        out.append((i * 3.0, (lambda b=jpg: b)))
    return out


@pytest.fixture
def setup():
    clock = Clock()
    emu = EmulatedPrinter(Playback(session(), clock=clock), serial="S1", check_code="C", name="p1").start()
    spec = PrinterSpec("p1", "127.0.0.1", "S1", "C", api_port=emu.api_port,
                       camera=f"http://127.0.0.1:{emu.camera_port}/?action=snapshot")
    mon = Monitor([spec], FakeDetector(), ServiceConfig(action="none", watch=WatchConfig(window=3, min_hits=2)), timeout=2)
    yield clock, emu, mon
    emu.stop()


def test_state_frame_and_manual_control(setup):
    clock, emu, mon = setup
    mon.step(now=0)
    client = create_app(mon).test_client()
    s = client.get("/api/state").get_json()
    p = s["printers"][0]
    assert p["name"] == "p1" and p["status"] == "printing" and p["has_frame"] and p["progress"] is not None
    r = client.get("/api/frame/p1.jpg")
    assert r.status_code == 200 and r.mimetype == "image/jpeg" and r.data[:2] == b"\xff\xd8"
    assert client.get("/api/frame/nope.jpg").status_code == 404

    r = client.post("/api/printer/p1/pause")
    assert r.status_code == 200 and r.get_json()["ok"]
    assert emu.playback.status == "paused"
    r = client.post("/api/printer/p1/resume")
    assert r.get_json()["ok"] and emu.playback.status == "printing"
    assert client.post("/api/printer/p1/explode").status_code == 400
    kinds = [e["kind"] for e in client.get("/api/state").get_json()["events"]]
    assert kinds.count("manual") == 2


def test_problems_and_alert_shown(setup):
    clock, emu, mon = setup
    for k in range(9):
        clock.t = k * 3.0
        mon.step(now=clock.t)
    p = mon.state()["printers"][0]
    assert p["problems"] == ["spaghetti"] and p["last_alert"] == "spaghetti"


def test_token_required_for_control(setup):
    _, emu, mon = setup
    client = create_app(mon, token="secret").test_client()
    assert client.post("/api/printer/p1/pause").status_code == 403
    assert emu.playback.status == "printing"
    assert client.post("/api/printer/p1/pause?token=secret").status_code == 200
    assert client.get("/api/state").status_code == 200          # смотреть можно без токена


def test_page_served(setup):
    _, _, mon = setup
    r = create_app(mon).test_client().get("/")
    assert r.status_code == 200 and "Парк AD5M" in r.get_data(as_text=True)
