import pytest

from ddet.alerts import AlertTracker


def test_single_hit_does_not_alert():
    t = AlertTracker(window=5, min_hits=3, cooldown_s=60)
    assert t.update({"spaghetti"}, 0) == []
    assert t.update(set(), 1) == []


def test_alert_after_min_hits_in_window():
    t = AlertTracker(window=5, min_hits=3, cooldown_s=60)
    assert t.update({"spaghetti"}, 0) == []
    assert t.update(set(), 1) == []
    assert t.update({"spaghetti"}, 2) == []
    assert t.update({"spaghetti"}, 3) == ["spaghetti"]


def test_old_hits_fall_out_of_window():
    t = AlertTracker(window=3, min_hits=2, cooldown_s=0)
    t.update({"x"}, 0)
    t.update(set(), 1)
    t.update(set(), 2)
    assert t.update({"x"}, 3) == []  # первое попадание уже выпало из окна


def test_cooldown_suppresses_repeat():
    t = AlertTracker(window=2, min_hits=1, cooldown_s=60)
    assert t.update({"x"}, 0) == ["x"]
    assert t.update({"x"}, 30) == []
    assert t.update({"x"}, 61) == ["x"]


def test_classes_independent():
    t = AlertTracker(window=2, min_hits=1, cooldown_s=60)
    assert t.update({"a", "b"}, 0) == ["a", "b"]
    assert t.update({"a", "c"}, 1) == ["c"]


def test_invalid_params():
    with pytest.raises(ValueError):
        AlertTracker(window=2, min_hits=3, cooldown_s=0)
