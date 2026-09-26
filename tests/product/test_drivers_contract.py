"""Контрактные тесты: одни и те же проверки для каждого протокола против эмулятора AD5M."""
import itertools

import pytest

from emulator.servers import PrinterNode
from emulator.virtual import VirtualPrinter, synthetic_frames
from guard.adapters.camera import HttpCamera
from guard.adapters.drivers import make_driver, probe
from guard.domain.models import PrinterConfig, Protocol, Status
from guard.ports import AuthError, PrinterError

_hosts = itertools.count(20)


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def node(kind: str, clock: Clock) -> PrinterNode:
    p = VirtualPrinter(f"emu-{kind}", "SNEMU0001", "12345678", kind=kind, heat_s=6, clock=clock)
    return PrinterNode(p, host=f"127.0.0.{next(_hosts)}", ports={"http": 0, "tcp": 0, "moonraker": 0, "camera": 0}).start()


def config(n: PrinterNode, variant: str, **kw) -> PrinterConfig:
    base = dict(name="p", host=n.host, camera_url=f"http://{n.host}:{n.port('camera')}/?action=snapshot")
    if variant == "moonraker":
        base |= dict(protocol="moonraker", moonraker_port=n.port("moonraker"))
    else:
        base |= dict(protocol="flashforge", http_port=n.port("http"), tcp_port=n.port("tcp"))
        if variant == "ff_http":
            base |= dict(serial="SNEMU0001", check_code="12345678")
    return PrinterConfig(**(base | kw))


VARIANTS = ["ff_http", "ff_tcp", "moonraker"]


@pytest.fixture(params=VARIANTS)
def env(request):
    clock = Clock()
    n = node("moonraker" if request.param == "moonraker" else "flashforge", clock)
    yield request.param, n, clock
    n.stop()


def test_idle_then_print_lifecycle(env):
    variant, n, clock = env
    d = make_driver(config(n, variant))
    try:
        assert d.state().status is Status.IDLE
        n.printer.start(synthetic_frames(40, 3.0), "cube.gcode", total_layers=100)
        clock.t = 30                                   # нагрев 6 с позади, печать идёт 24 с
        s = d.state()
        assert s.status is Status.PRINTING
        assert 0.15 < s.progress < 0.25 and s.file == "cube.gcode"
        assert s.nozzle_target == 220 and s.bed_target == 60 and 15 <= s.layer <= 25
        d.pause()
        assert n.printer.status == "paused" and d.state().status is Status.PAUSED
        d.resume()
        assert d.state().status is Status.PRINTING
        d.cancel()
        assert d.state().status in (Status.CANCELLED, Status.IDLE)   # TCP сообщает отмену как READY
        clock.t = 10_000
    finally:
        d.close()


def test_completed(env):
    variant, n, clock = env
    d = make_driver(config(n, variant))
    try:
        n.printer.start(synthetic_frames(10, 3.0))
        clock.t = 1000
        s = d.state()
        assert s.status is Status.COMPLETED and s.progress == pytest.approx(1.0)
    finally:
        d.close()


def test_offline_is_printer_error(env):
    variant, n, clock = env
    d = make_driver(config(n, variant), timeout=1.0)
    n.printer.faults.offline = True
    try:
        with pytest.raises(PrinterError):
            d.state()
    finally:
        d.close()


def test_probe_reports_identity(env):
    variant, n, _ = env
    cfg = config(n, variant, protocol="auto")
    info = probe(cfg)
    if variant == "moonraker":
        assert info.protocol is Protocol.MOONRAKER and info.name == "emu-moonraker"
        assert info.camera_url.endswith(":8080/?action=snapshot")
    else:
        assert info.protocol is Protocol.FLASHFORGE and info.serial == "SNEMU0001" and "5M" in info.model


def test_camera_snapshot_stream_and_down(env):
    _, n, _ = env
    cam = HttpCamera(f"http://{n.host}:{n.port('camera')}/?action=snapshot", timeout=2)
    assert cam.snapshot()[:2] == b"\xff\xd8"
    stream = HttpCamera(f"http://{n.host}:{n.port('camera')}/?action=stream", timeout=2)
    assert stream.snapshot()[:2] == b"\xff\xd8"
    n.printer.faults.camera_down = True
    with pytest.raises(PrinterError):
        cam.snapshot()


# ------------------------------------------------------------------ особенности стоковой прошивки
@pytest.fixture
def ff():
    clock = Clock()
    n = node("flashforge", clock)
    yield n, clock
    n.stop()


def test_wrong_code_and_lan_mode_are_auth_errors(ff):
    n, _ = ff
    d = make_driver(config(n, "ff_http", check_code="wrong"))
    with pytest.raises(AuthError):
        d.state()
    n.printer.faults.lan_only = False
    d2 = make_driver(config(n, "ff_http"))
    with pytest.raises(AuthError, match="Только LAN"):
        d2.state()
    d.close(), d2.close()


def test_probe_fills_serial_for_http(ff):
    n, _ = ff
    cfg = config(n, "ff_tcp", check_code="12345678", protocol="auto")
    assert cfg.serial == ""
    probe(cfg)
    assert cfg.serial == "SNEMU0001"


def test_control_falls_back_to_tcp_when_http_down(ff):
    n, clock = ff
    n.printer.start(synthetic_frames(40, 3.0))
    clock.t = 30
    cfg = config(n, "ff_http", http_port=1)       # HTTP «лёг»
    d = make_driver(cfg, timeout=1.0)
    try:
        assert d.state().status is Status.PRINTING  # состояние — по TCP
        d.pause()
        assert n.printer.status == "paused"
    finally:
        d.close()


def test_tcp_single_session(ff):
    n, _ = ff
    a, b = make_driver(config(n, "ff_tcp")), make_driver(config(n, "ff_tcp"), timeout=1.0)
    try:
        a.state()
        with pytest.raises(PrinterError, match="сессию"):
            b.state()
    finally:
        a.close(), b.close()


def test_firmware_error_reported(env):
    variant, n, _ = env
    if variant == "ff_tcp":
        pytest.skip("TCP не сообщает код ошибки")
    n.printer.faults.firmware_error = "E0017"
    d = make_driver(config(n, variant))
    try:
        s = d.state()
        assert s.status is Status.ERROR and "E0017" in (s.message or "")
    finally:
        d.close()


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PrinterConfig(name="x", host="bad host!")
    with pytest.raises(ValueError):
        PrinterConfig(name="", host="1.2.3.4")
    with pytest.raises(ValueError):
        PrinterConfig(name="x", host="1.2.3.4", camera_url="ftp://cam")
