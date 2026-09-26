import zipfile

import cv2
import numpy as np

from ddet.detect import Detection
from ddet.sources import open_source
from ddet.watch import NO_PLATE, PrintWatcher, WatchConfig

BOX = (0, 0, 10, 10)
PLATE = Detection("scene", "pei_plate", 0.9, BOX)


def d(model, cls, conf=0.9):
    return Detection(model, cls, conf, BOX)


def cfg(**kw):
    return WatchConfig(window=3, min_hits=2, cooldown_s=100, hand_hold_s=10, **kw)


def test_defect_alerts_after_repeated_hits():
    w = PrintWatcher(cfg(), has_scene=False)
    assert w.update([d("defects", "spaghetti")], 0)[1] == []
    assert w.update([d("defects", "spaghetti")], 1)[1] == ["spaghetti"]


def test_low_confidence_ignored():
    w = PrintWatcher(cfg(defect_conf=0.5), has_scene=False)
    for t in range(5):
        found, fired = w.update([d("defects", "spaghetti", 0.3)], t)
        assert not found and not fired


def test_missing_plate_only_with_scene_model():
    assert NO_PLATE in PrintWatcher(cfg(), has_scene=True).problems([])
    assert NO_PLATE not in PrintWatcher(cfg(), has_scene=True).problems([PLATE])
    assert NO_PLATE not in PrintWatcher(cfg(), has_scene=False).problems([])


def test_scene_problem_classes_raise_alert_but_plate_and_head_do_not():
    w = PrintWatcher(cfg(), has_scene=True)
    dets = [PLATE, d("scene", "printer_head"), d("scene", "pei_misplaced")]
    assert w.problems(dets) == {"pei_misplaced"}


def test_glass_instead_of_pei_is_a_problem_but_plate_is_visible():
    w = PrintWatcher(cfg(), has_scene=True)
    assert w.problems([d("scene", "glass_plate")]) == {"glass_plate"}


def test_muted_class_is_seen_but_never_alerts():
    w = PrintWatcher(cfg(), has_scene=True)
    for t in range(5):
        found, fired = w.update([PLATE, d("scene", "pei_misplaced")], t)
        assert found == {"pei_misplaced"} and fired == []


def test_hand_suppresses_judgement_for_a_while():
    w = PrintWatcher(cfg(), has_scene=True)
    bad = [PLATE, d("defects", "spaghetti")]
    assert w.update(bad + [d("scene", "hand")], 0) == (set(), [])
    assert w.update(bad, 5) == (set(), [])          # ещё 10 с после руки
    assert w.update(bad, 11)[1] == []               # первое попадание
    assert w.update(bad, 12)[1] == ["spaghetti"]


def _jpeg(v):
    ok, buf = cv2.imencode(".jpg", np.full((8, 8, 3), v, np.uint8))
    return buf.tobytes()


def test_frame_zip_ordered_by_timestamp_and_thinned(tmp_path):
    z = tmp_path / "s.zip"
    names = ["printer_20250101_100009.jpg", "printer_20250101_100000.jpg",
             "printer_20250101_100003.jpg", "printer_20250101_100006.jpg"]
    with zipfile.ZipFile(z, "w") as f:
        for i, n in enumerate(names):
            f.writestr(f"x/{n}", _jpeg(i * 10))
    frames = list(open_source(str(z), every=5))
    assert [fr.t for fr in frames] == [0.0, 6.0]
    assert frames[0].label == "2025-01-01 10:00:00"


def test_video_file_sampling(tmp_path):
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 32))
    for i in range(30):
        w.write(np.full((32, 32, 3), i * 8, np.uint8))
    w.release()
    frames = list(open_source(str(path), every=1.0))
    assert [fr.index for fr in frames] == [0, 10, 20]
    assert [fr.t for fr in frames] == [0.0, 1.0, 2.0]
