import pytest

from fakes import FakeCamera, FakeDetector, FakeDriver, FakePrinter, MemConfig, MemEvents
from guard.domain.models import PrinterConfig, Status
from guard.domain.policy import Settings
from guard.services.fleet import FleetService


def make(n=1, settings=None, detector_cls="spaghetti"):
    printers = {}
    cfgs = [PrinterConfig(name=f"p{i}", host=f"10.0.0.{i + 1}", protocol="flashforge") for i in range(n)]
    for c in cfgs:
        printers[c.id] = FakePrinter()
    store, events = MemConfig(settings or Settings(window=3, min_hits=2, baseline_s=0), cfgs), MemEvents()
    svc = FleetService(store, events, FakeDetector(detector_cls),
                       driver_factory=lambda c: FakeDriver(printers[c.id]),
                       camera_factory=lambda c: FakeCamera(printers[c.id]))
    return svc, printers, cfgs, events, store


def run(svc, cycles, t0=0.0):
    for k in range(cycles):
        svc.step(now=t0 + k * 5)


def test_clean_print_never_touched():
    svc, printers, cfgs, events, _ = make()
    run(svc, 20)
    assert printers[cfgs[0].id].commands == []
    assert "alert" not in events.kinds()


def test_spaghetti_confirmed_then_paused_once():
    svc, printers, cfgs, events, _ = make()
    p = printers[cfgs[0].id]
    run(svc, 2)
    p.bright = True
    run(svc, 2, t0=100)
    assert p.commands == []                       # первый кадр — только место, второй — одно подтверждение
    run(svc, 1, t0=110)
    assert p.commands == ["pause"] and p.status is Status.PAUSED
    assert events.kinds().count("stopped") == 1
    run(svc, 5, t0=110)                           # на паузе не судим и не жмём паузу повторно
    assert p.commands == ["pause"]


def test_notify_class_alerts_without_stop():
    svc, printers, cfgs, events, _ = make(settings=Settings(window=3, min_hits=2, notify_on=["garbage"], baseline_s=0),
                                           detector_cls="garbage")
    p = printers[cfgs[0].id]
    p.bright = True
    run(svc, 5)
    assert p.commands == [] and "alert" in events.kinds() and "stopped" not in events.kinds()


def test_silent_class_visible_but_no_alarm():
    svc, printers, cfgs, events, _ = make(detector_cls="garbage")      # по умолчанию мусор — тихий
    printers[cfgs[0].id].bright = True
    run(svc, 5)
    assert "alert" not in events.kinds() and printers[cfgs[0].id].commands == []
    assert svc.view()["printers"][0]["problems"] == ["garbage"]


def test_action_none_only_alerts():
    svc, printers, cfgs, events, _ = make(settings=Settings(window=3, min_hits=2, action="none"))
    printers[cfgs[0].id].bright = True
    run(svc, 5)
    assert printers[cfgs[0].id].commands == [] and "alert" in events.kinds()


def test_only_bad_printer_in_fleet_stopped():
    svc, printers, cfgs, _, _ = make(n=4)
    printers[cfgs[2].id].bright = True
    run(svc, 5)
    assert [printers[c.id].commands for c in cfgs] == [[], [], ["pause"], []]


def test_offline_reported_once_and_recovery():
    svc, printers, cfgs, events, _ = make()
    p = printers[cfgs[0].id]
    p.offline = True
    run(svc, 4)
    assert events.kinds().count("error") == 1
    assert svc.view()["printers"][0]["state"]["status"] == "offline"
    p.offline = False
    run(svc, 1, t0=100)
    assert "снова на связи" in [e.detail for e in events.items]


def test_camera_down_does_not_mark_printer_offline():
    svc, printers, cfgs, events, _ = make()
    printers[cfgs[0].id].camera_down = True
    run(svc, 3)
    v = svc.view()["printers"][0]
    assert v["state"]["status"] == "printing" and "камера" in v["camera_error"]


def test_manual_resume_resets_history():
    svc, printers, cfgs, events, _ = make()
    p = printers[cfgs[0].id]
    p.bright = True
    run(svc, 3)
    assert p.status is Status.PAUSED
    p.bright = False
    svc.control(cfgs[0].id, "resume")
    run(svc, 5, t0=100)
    assert p.commands == ["pause", "resume"] and p.status is Status.PRINTING


def test_detector_failure_does_not_crash_cycle():
    svc, printers, cfgs, events, _ = make()

    class Boom:
        has_scene = False

        def detect(self, images):
            raise RuntimeError("onnx упал")
    svc.detector = Boom()
    run(svc, 2)
    assert any("распознавание не удалось" in e.detail for e in events.items)


def test_crud_persists_and_validates():
    svc, printers, cfgs, events, store = make(n=1)
    cfg = svc.add_printer({"name": "new", "host": "192.168.0.50", "protocol": "moonraker"})
    assert store.saves == 1 and any(p.name == "new" for p in store.printers)
    with pytest.raises(ValueError):
        svc.add_printer({"name": "new", "host": "192.168.0.51"})           # имя занято
    with pytest.raises(ValueError):
        svc.add_printer({"name": "x", "host": "not a host"})
    svc.update_printer(cfg.id, {"host": "192.168.0.60", "check_code": "••••"})
    assert svc.workers[cfg.id].cfg.host == "192.168.0.60"
    svc.remove_printer(cfg.id)
    assert cfg.id not in svc.workers and not any(p.name == "new" for p in store.printers)


def test_settings_update_validates_and_keeps_secret():
    svc, *_ , store = make()
    svc.update_settings({"telegram_token": "secret", "defect_conf": 0.7})
    svc.update_settings({"telegram_token": "••••", "action": "cancel"})
    assert store.settings.telegram_token == "secret" and store.settings.action == "cancel"
    with pytest.raises(ValueError):
        svc.update_settings({"action": "explode"})
    with pytest.raises(ValueError):
        svc.update_settings({"window": 2, "min_hits": 3})


def test_disabled_printer_not_polled():
    svc, printers, cfgs, events, _ = make()
    svc.update_printer(cfgs[0].id, {"enabled": False})
    printers[cfgs[0].id].bright = True
    run(svc, 5)
    assert printers[cfgs[0].id].commands == [] and svc.detector.calls == 0


def test_alert_saves_raw_frames_for_training(tmp_path):
    import json
    from guard.adapters.storage import SqliteEventStore
    svc, printers, cfgs, _, _ = make(settings=Settings(window=3, min_hits=2, baseline_s=0, training_frames=4))
    svc.events = SqliteEventStore(tmp_path / "e.db", tmp_path / "alerts")
    run(svc, 3)
    printers[cfgs[0].id].bright = True
    run(svc, 4, t0=100)
    [folder] = list((tmp_path / "training").iterdir())
    assert sorted(p.name for p in folder.glob("*.jpg")) == ["00.jpg", "01.jpg", "02.jpg", "03.jpg", "04_alert.jpg"]
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["fired"] == ["spaghetti"] and len(meta["frames"]) == 5 and meta["frames"][-1]["detections"]
    import zipfile
    info = svc.events.export_training(tmp_path / "out.zip", 30, {"app": "test"})
    names = zipfile.ZipFile(tmp_path / "out.zip").namelist()
    assert info["alerts"] == 1 and "info.json" in names and f"{folder.name}/04_alert.jpg" in names


def test_manual_training_snapshot(tmp_path):
    from guard.adapters.storage import SqliteEventStore
    svc, printers, cfgs, _, _ = make()
    svc.events = SqliteEventStore(tmp_path / "e.db", tmp_path / "alerts")
    run(svc, 3)
    name = svc.training_snapshot(cfgs[0].id, "пропущен брак")
    folder = tmp_path / "training" / name
    assert (folder / "meta.json").exists() and list(folder.glob("*_alert.jpg"))
    assert "пропущен брак" in (folder / "meta.json").read_text(encoding="utf-8")
