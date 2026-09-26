import zipfile
from datetime import datetime, timedelta

import cv2
import numpy as np

import extract_frames
from extract_frames import Candidate, cap, scan, select

T0 = datetime(2025, 3, 1, 12, 0, 0)


def cand(sec, value=0, score=0.0):
    c = Candidate(T0 + timedelta(seconds=sec), f"f{sec}.jpg", lambda: b"")
    c.thumb = np.full((60, 80), value, np.int16)
    c.score = score
    return c


def test_select_keeps_changes_and_periodic_frames():
    # Статичная сцена 0..300 с, в 90 с сцена меняется.
    session = [cand(s, 50 if s >= 90 else 0) for s in range(0, 301, 3)]
    kept = select(session, min_gap=20, max_gap=120, change=4)
    secs = [int((c.taken - T0).total_seconds()) for c in kept]
    assert secs[0] == 0
    assert 90 in secs                      # заметное изменение
    assert all(b - a >= 20 for a, b in zip(secs, secs[1:]))
    assert max(b - a for a, b in zip(secs, secs[1:])) <= 120


def test_select_skips_non_camera_frames():
    other_camera = cand(0)
    other_camera.thumb = None
    ad5m = cand(30)
    assert select([other_camera, ad5m], 20, 600, 4) == [ad5m]


def test_cap_prefers_suspicious_and_spreads_rest():
    kept = [cand(s * 30, score=0.9 if s == 7 else 0.0) for s in range(40)]
    picked = cap(kept, 9)
    assert len(picked) <= 9
    assert any(c.score == 0.9 for c in picked)
    secs = [(c.taken - T0).total_seconds() for c in picked]
    assert secs == sorted(secs) and secs[0] == 0 and secs[-1] == 39 * 30


def _jpeg(size):
    ok, buf = cv2.imencode(".jpg", np.zeros((size[1], size[0], 3), np.uint8))
    return buf.tobytes()


def test_scan_dedupes_copies_and_reads_zips(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "printer_20250301_120000.jpg").write_bytes(_jpeg((640, 480)))
    with zipfile.ZipFile(tmp_path / "b.zip", "w") as z:
        z.writestr("x/printer_20250301_120000.jpg", _jpeg((640, 480)))       # копия того же кадра
        z.writestr("x/printer_20250301_120003.jpg", _jpeg((640, 480)))
        z.writestr("x/output_20250301_120003.jpg", _jpeg((640, 480)))        # не кадр грабера
    (tmp_path / "skip").mkdir()
    (tmp_path / "skip" / "printer_20250301_130000.jpg").write_bytes(_jpeg((640, 480)))
    found = scan([tmp_path], exclude=[tmp_path / "skip"])
    assert sorted(found) == ["printer_20250301_120000", "printer_20250301_120003"]


def test_thumbnail_rejects_other_cameras():
    assert extract_frames.thumbnail(_jpeg((640, 480))) is not None
    assert extract_frames.thumbnail(_jpeg((1920, 1080))) is None
