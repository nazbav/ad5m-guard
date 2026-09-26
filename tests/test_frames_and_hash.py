from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw

from ddet.frames import parse_timestamp, split_sessions
from ddet.imagehash import UnionFind, dhash, near_duplicate_pairs, pack


def test_timestamp_from_grabber_and_roboflow_names():
    assert parse_timestamp("printer_20250309_154916.jpg") == datetime(2025, 3, 9, 15, 49, 16)
    assert parse_timestamp("printer_20241230_193218_jpg.rf.abc123.jpg") == datetime(2024, 12, 30, 19, 32, 18)
    assert parse_timestamp("WhatsApp-Image-2024.jpg") is None
    assert parse_timestamp("printer_20251399_999999.jpg") is None  # невозможная дата


def test_sessions_split_on_gap():
    t0 = datetime(2025, 1, 1, 10, 0)
    times = [t0, t0 + timedelta(minutes=1), t0 + timedelta(minutes=5), t0 + timedelta(hours=2)]
    sessions = split_sessions([(t, i) for i, t in enumerate(reversed(times))], timedelta(minutes=20))
    assert [len(s) for s in sessions] == [3, 1]
    assert sessions[0][0][0] == t0  # отсортировано по времени


def _img(shift=0, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    img = Image.fromarray(np.roll(a, shift, axis=1))
    return img


def test_near_duplicates_found_and_different_images_not():
    base = _img()
    recompressed = base.resize((320, 240)).resize((160, 120))
    other = _img(seed=7)
    hashes = pack([dhash(base), dhash(recompressed), dhash(other)])
    pairs = set(near_duplicate_pairs(hashes, max_distance=12))
    assert (0, 1) in pairs
    assert (0, 2) not in pairs and (1, 2) not in pairs


def test_union_find_transitive():
    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.find(0) == uf.find(2) != uf.find(3)
