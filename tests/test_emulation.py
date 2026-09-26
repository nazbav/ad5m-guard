"""Работа сервиса против эмулированных принтеров AD5M по настоящему HTTP.

Печать синтетическая: кадры каждые 3 с, с кадра DEFECT_FROM картинка светлеет —
поддельный детектор считает светлый кадр спагетти. Часы подменены, поэтому
тесты детерминированы и идут мгновенно.
"""
import cv2
import numpy as np
import pytest

from ddet.detect import Detection
from ddet.emulator import EmulatedPrinter, Playback
from ddet.printer import AD5MClient, PrinterError, PrinterSpec
from ddet.service import Monitor, ServiceConfig
from ddet.sources import open_source
from ddet.watch import WatchConfig

N_FRAMES, STEP_S, DEFECT_FROM = 40, 3.0, 20


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeDetector:
    models = {"defects": None}

    def batch(self, images):
        return [[Detection("defects", "spaghetti", 0.9, (10, 10, 50, 50))] if img.mean() > 128 else []
                for img in images]


def session(defect: bool):
    frames = []
    for i in range(N_FRAMES):
        value = 220 if defect and i >= DEFECT_FROM else 30
        jpg = cv2.imencode(".jpg", np.full((480, 640, 3), value, np.uint8))[1].tobytes()
        frames.append((i * STEP_S, (lambda b=jpg: b)))
    return frames


@pytest.fixture
def fleet():
    started = []
    clock = Clock()

    def make(defect: bool, name: str, check_code="1234"):
        emu = EmulatedPrinter(Playback(session(defect), clock=clock), serial=f"SN-{name}", check_code="1234",
                              name=name).start()
        started.append(emu)
        spec = PrinterSpec(name, "127.0.0.1", f"SN-{name}", check_code, api_port=emu.api_port,
                           camera=f"http://127.0.0.1:{emu.camera_port}/?action=snapshot")
        return emu, spec

    yield clock, make
    for emu in started:
        emu.stop()


def monitor(specs, action="pause"):
    return Monitor(specs, FakeDetector(), ServiceConfig(action=action, watch=WatchConfig(window=5, min_hits=3)),
                   timeout=2)


def play(clock, mon, until_frame=N_FRAMES + 2):
    events = []
    for k in range(until_frame):
        clock.t = k * STEP_S
        events += mon.step(now=clock.t)
    return events


def test_defective_print_is_paused_right_after_defect_confirmed(fleet):
    clock, make = fleet
    emu, spec = make(defect=True, name="p1")
    events = play(clock, monitor([spec]))
    assert emu.playback.status == "paused"
    assert emu.playback.commands == ["pause"]
    # брак с кадра 20, тревога — на третьем подряд подтверждении
    assert emu.playback.frame_index() == DEFECT_FROM + 2
    kinds = [e.kind for e in events]
    assert kinds.count("alert") == 1 and kinds.count("stopped") == 1


def test_clean_print_is_never_touched(fleet):
    clock, make = fleet
    emu, spec = make(defect=False, name="p1")
    events = play(clock, monitor([spec]))
    assert emu.playback.commands == []
    assert emu.playback.current_status() == "completed"
    assert not [e for e in events if e.kind in ("alert", "stopped")]


def test_only_defective_printer_in_fleet_is_stopped(fleet):
    clock, make = fleet
    printers = [make(defect=(i == 1), name=f"p{i}") for i in range(4)]
    play(clock, monitor([spec for _, spec in printers]))
    assert [emu.playback.commands for emu, _ in printers] == [[], ["pause"], [], []]


def test_notify_only_class_alerts_without_stopping(fleet):
    clock, make = fleet
    emu, spec = make(defect=True, name="p1")
    mon = monitor([spec])
    mon.detector = GarbageDetector()
    events = play(clock, mon)
    assert emu.playback.commands == []
    assert [e.detail for e in events if e.kind == "alert"] == ["garbage"]


class GarbageDetector(FakeDetector):
    def batch(self, images):
        return [[Detection("defects", "garbage", 0.9, (10, 10, 50, 50))] if img.mean() > 128 else []
                for img in images]


def test_cancel_action(fleet):
    clock, make = fleet
    emu, spec = make(defect=True, name="p1")
    play(clock, monitor([spec], action="cancel"))
    assert emu.playback.status == "cancel"


def test_action_none_only_alerts(fleet):
    clock, make = fleet
    emu, spec = make(defect=True, name="p1")
    events = play(clock, monitor([spec], action="none"))
    assert emu.playback.commands == []
    assert any(e.kind == "alert" for e in events)


def test_wrong_check_code_reported_once_and_does_not_crash(fleet):
    clock, make = fleet
    emu, spec = make(defect=True, name="p1", check_code="wrong")
    events = play(clock, monitor([spec]), until_frame=5)
    errors = [e for e in events if e.kind == "error"]
    assert len(errors) == 1 and "Unauthorized" in errors[0].detail
    assert emu.playback.commands == []


def test_not_judged_while_paused_and_history_reset_on_resume(fleet):
    clock, make = fleet
    emu, spec = make(defect=True, name="p1")
    mon = monitor([spec])
    events = play(clock, mon, until_frame=DEFECT_FROM + 3)       # приостановлен на кадре 22
    assert emu.playback.status == "paused"
    for k in range(5):                                           # стоит на паузе — не трогаем
        clock.t += STEP_S
        events += mon.step(now=clock.t)
    assert emu.playback.commands == ["pause"]
    AD5MClient(spec).resume()                                    # человек продолжил печать
    for k in range(2):                                           # две проверки — мало для новой тревоги
        clock.t += STEP_S
        mon.step(now=clock.t)
    assert emu.playback.status == "printing"
    clock.t += STEP_S
    mon.step(now=clock.t)                                        # третья подряд — снова пауза
    assert emu.playback.commands == ["pause", "continue", "pause"]


def test_unreachable_printer_reported(fleet):
    clock, _ = fleet
    spec = PrinterSpec("ghost", "127.0.0.1", "SN", "0", api_port=1)
    events = monitor([spec]).step(now=0)
    assert events and events[0].kind == "error"


def test_client_rejects_invalid_state():
    clock = Clock()
    emu = EmulatedPrinter(Playback(session(False), clock=clock), serial="S", check_code="C").start()
    try:
        client = AD5MClient(PrinterSpec("x", "127.0.0.1", "S", "C", api_port=emu.api_port))
        assert client.status() == "printing"
        client.pause()
        with pytest.raises(PrinterError):
            client.pause()          # уже на паузе
    finally:
        emu.stop()


def test_playback_timeline():
    clock = Clock()
    pb = Playback(session(False), speed=2.0, clock=clock)
    clock.t = 15                      # ×2 → 30 с записи → кадр 10
    assert pb.frame_index() == 10
    pb.control("pause")
    clock.t = 100
    assert pb.frame_index() == 10     # на паузе кадр замер
    pb.control("continue")
    clock.t = 1000
    assert pb.current_status() == "completed" and pb.frame_index() == N_FRAMES - 1


def test_mjpeg_stream_readable_like_real_camera():
    emu = EmulatedPrinter(Playback(session(False)), serial="S", check_code="C").start()
    try:
        frames = open_source(f"http://127.0.0.1:{emu.camera_port}/?action=stream", every=0.1)
        first = next(frames)
        assert first.image.shape == (480, 640, 3)
        frames.close()
    finally:
        emu.stop()


def test_example_config_loads():
    import monitor
    from ddet.config import CONFIG_DIR
    cfg, specs = monitor.load_config(CONFIG_DIR / "printers.example.yaml")
    assert cfg.action == "pause" and cfg.watch.min_hits == 3 and cfg.watch.cooldown_s == 600
    assert cfg.watch.notify_only == {"garbage", "stringing"} and cfg.watch.muted == {"pei_misplaced"}
    assert specs[0].camera_url == "http://192.168.0.108:8080/?action=snapshot"
