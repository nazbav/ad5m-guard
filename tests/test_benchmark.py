from datetime import datetime, timedelta

import cv2
import numpy as np

import benchmark
from ddet.detect import Detection
from guard.domain.policy import Settings

T0 = datetime(2025, 3, 4, 20, 0, 0)


def jpg(v):
    return cv2.imencode(".jpg", np.full((48, 64, 3), v, np.uint8))[1].tobytes()


class FakeDetector:
    """Светлый кадр — спагетти, серый — мусор (только уведомление), тёмный — ничего."""
    models = {"defects": None}

    def batch(self, images):
        out = []
        for img in images:
            m = img.mean()
            if m > 200:
                out.append([Detection("defects", "spaghetti", 0.9, (1, 1, 10, 10))])
            elif m > 100:
                out.append([Detection("defects", "garbage", 0.9, (1, 1, 10, 10))])
            else:
                out.append([])
        return out


def frames(values, step=10):
    return [(T0 + timedelta(seconds=i * step), (lambda b=jpg(v): b)) for i, v in enumerate(values)]


def test_sample_every():
    fr = frames([0] * 10, step=3)
    assert [ts.second for ts, _ in benchmark.sample(fr, 10)] == [0, 12, 24]


def test_failure_window_catch_and_false_stop(tmp_path):
    # 0–100 с тёмные, 100–160 с спагетти (ожидаемый сбой), 300–360 с снова спагетти (ложная), мусор не в счёт
    values = [0] * 10 + [255] * 6 + [0] * 8 + [150] * 6 + [255] * 6 + [0] * 4
    fail = benchmark.Failure("s", T0 + timedelta(seconds=100), T0 + timedelta(seconds=160), "spaghetti", "")
    cfg = Settings(window=5, min_hits=3, cooldown_s=100, spatial=False)
    r = benchmark.run_session("s", frames(values), FakeDetector(), cfg, [fail], timedelta(seconds=30), tmp_path)
    assert fail.detected_at == T0 + timedelta(seconds=120)          # третья проверка подряд
    assert [cls for _, cls in r.false_stops] == ["spaghetti"]
    assert len(list((tmp_path / "false_stops").glob("*.jpg"))) == 1
    assert r.hours > 0


def test_plate_point_inside_and_degenerate():
    from ddet.synth import plate_point
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = plate_point((0, 200, 640, 480), rng)
        assert 40 <= x <= 600 and 200 <= y <= 420
    assert plate_point((0, 470, 640, 480), rng) == (320, 475)      # только полоска у нижнего края


def test_distractor_any_size():
    from ddet.synth import distractor
    rng = np.random.default_rng(0)
    tex = np.zeros((480, 640, 3), np.uint8)
    for h, w in [(10, 10), (400, 600), (479, 639), (700, 900)]:
        cut = np.zeros((h, w, 4), np.uint8)
        assert distractor(cut, tex, rng).shape == (h, w, 4)
