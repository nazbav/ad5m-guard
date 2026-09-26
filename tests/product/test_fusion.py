import pytest

from guard.domain.fusion import fuse_defects
from guard.domain.models import Detection


def d(cls, conf, x=100, model="defects"):
    return Detection(model, cls, conf, (x, 100, x + 60, 160))


def confs(dets):
    return sorted((x.cls, round(x.conf, 2)) for x in dets)


def test_and_needs_both_models_same_place():
    assert confs(fuse_defects([d("spaghetti", 0.9)], [d("spaghetti", 0.6, x=110)], "and")) == [("spaghetti", 0.6)]
    assert fuse_defects([d("spaghetti", 0.9)], [], "and") == []
    assert fuse_defects([d("spaghetti", 0.9)], [d("spaghetti", 0.9, x=400)], "and") == []      # другое место
    assert fuse_defects([d("spaghetti", 0.9)], [d("garbage", 0.9)], "and") == []               # другой класс


def test_or_takes_any():
    assert confs(fuse_defects([d("spaghetti", 0.9)], [d("garbage", 0.5, x=400)], "or")) == [("garbage", 0.5), ("spaghetti", 0.9)]


def test_avg_counts_missing_as_zero():
    fused = fuse_defects([d("spaghetti", 0.8)], [d("spaghetti", 0.6, x=105), d("detached", 0.8, x=400)], "avg")
    assert confs(fused) == [("detached", 0.4), ("spaghetti", 0.7)]


def test_each_detection_matched_once_and_output_is_defects_model():
    fused = fuse_defects([d("spaghetti", 0.9), d("spaghetti", 0.8, x=102)], [d("spaghetti", 0.7)], "and")
    assert len(fused) == 1 and fused[0].model == "defects"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        fuse_defects([], [], "xor")
