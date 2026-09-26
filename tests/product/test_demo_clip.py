"""Поставляемый набор моделей (build/models + models.yaml) и рекомендуемое правило на кадрах демо
(настоящие записи AD5M): сбой должен остановить печать, чистая печать — ни разу."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from guard.domain.policy import AlertPolicy, Settings

ROOT = Path(__file__).resolve().parents[2]
MODELS, DEMO = ROOT / "build" / "models", ROOT / "build" / "demo"
pytestmark = pytest.mark.skipif(not (MODELS / "defects.onnx").exists() or not (DEMO / "spaghetti").exists(),
                                reason="нет build/models или build/demo")


def shipped_settings(**kw) -> Settings:
    from guard.app.main import model_manifest
    return Settings(**{**model_manifest(MODELS)["recommended"], **kw})


@pytest.fixture(scope="module")
def detections():
    from guard.app.main import build_detector
    det = build_detector(MODELS)
    det.focus = {"spaghetti", "detached", "plate_not_visible", "glass_plate"}
    out = {}
    for sub in ("clean", "clean2", "clean3", "spaghetti"):
        files = sorted((DEMO / sub).glob("*.jpg"))
        out[sub] = [det.detect([cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)])[0] for p in files]
    return out


def verdicts(seq, settings):
    pol = AlertPolicy(settings, has_scene=True)
    return [pol.update(d, k * settings.interval_s) for k, d in enumerate(seq)]


@pytest.mark.parametrize("clip", ["clean", "clean2", "clean3"])     # у каждого демо-принтера своя печать
def test_clean_print_never_stopped(detections, clip):
    assert not any(v.stop_for for v in verdicts(detections[clip], shipped_settings(interval_s=3)))


def test_demo_failure_stops_print(detections):
    """Как в демо на AD5M-01: 30 кадров чистой печати, затем ком нити."""
    vs = verdicts(detections["clean"][:30] + detections["spaghetti"], shipped_settings(interval_s=3))
    stops = [k - 30 for k, v in enumerate(vs) if v.stop_for]
    assert stops and 0 <= stops[0] < len(detections["spaghetti"]), stops      # до конца клипа, не раньше сбоя
