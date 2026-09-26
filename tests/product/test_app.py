"""Запуск из исходников берёт ресурсы из build/ (как exe — изнутри сборки) и реально грузит модели."""
import pytest

from guard.app import main


def test_resources_from_build_folder_in_sources():
    assert main.resource("models") == main.bundle_dir() / "build" / "models"
    assert main.resource("demo") == main.bundle_dir() / "build" / "demo"


@pytest.mark.skipif(not (main.resource("models") / "defects.onnx").exists(), reason="нет build/models")
def test_detector_loads_from_default_location():
    det = main.build_detector(main.resource("models"))
    assert det is not None and det.has_scene


def test_missing_models_disable_detection_not_crash(tmp_path):
    assert main.build_detector(tmp_path) is None


@pytest.mark.skipif(not (main.resource("models") / "defects_rfdetr.onnx").exists(), reason="нет второй модели")
def test_second_model_from_manifest_runs_only_when_first_sees_something(tmp_path):
    import shutil
    import cv2
    import numpy as np
    src = main.resource("models")
    for f in ("defects.onnx", "scene.onnx", "defects_rfdetr.onnx"):
        shutil.copy(src / f, tmp_path / f)
    (tmp_path / "models.yaml").write_text("second: defects_rfdetr.onnx\nfusion: and\ngate_conf: 0.3\n", encoding="utf-8")
    det = main.build_detector(tmp_path)
    assert "defects2" in det.models and det.fusion == "and"
    blank = np.full((480, 640, 3), 90, np.uint8)
    out = det.detect([blank])[0]
    assert det.second_runs == 0 and not [d for d in out if d.model == "defects"]
    demo = sorted((main.resource("demo") / "spaghetti").glob("*.jpg"))
    if demo:
        det.detect([cv2.imdecode(np.fromfile(str(demo[0]), np.uint8), cv2.IMREAD_COLOR)])
        assert det.second_runs == 1


def test_bad_manifest_disables_detection_not_crash(tmp_path):
    (tmp_path / "defects.onnx").write_bytes(b"not a model")
    (tmp_path / "models.yaml").write_text("fusion: [broken\n", encoding="utf-8")
    assert main.build_detector(tmp_path) is None


def test_single_instance_per_data_folder(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    first = main.single_instance(a)
    assert first is not None
    assert main.single_instance(a) is None            # вторая копия на те же данные — сразу видна
    assert main.single_instance(b) is not None        # демо / другая папка данных — можно


def test_second_server_cannot_take_same_port():
    import threading
    from werkzeug.serving import make_server
    main.exclusive_port()
    s1 = make_server("127.0.0.1", 0, lambda env, start: [], threaded=True)
    threading.Thread(target=s1.serve_forever, daemon=True).start()
    try:
        with pytest.raises((OSError, SystemExit)):       # werkzeug превращает отказ в sys.exit(1)
            make_server("127.0.0.1", s1.server_port, lambda env, start: [], threaded=True)
    finally:
        s1.shutdown()
