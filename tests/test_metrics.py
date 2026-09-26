import pytest

from ddet.metrics import frame_level, map50, per_class_frame_recall

GT = {"a.jpg": [[0, 10, 10, 50, 50]], "b.jpg": [[0, 100, 100, 150, 150], [1, 0, 0, 20, 20]], "c.jpg": []}


def test_perfect_predictions_give_ap_one():
    preds = {f: [[t[0], 0.9, *t[1:]] for t in v] for f, v in GT.items()}
    ap = map50(GT, preds, [0, 1, 2])
    assert ap[0] == pytest.approx(1.0) and ap[1] == pytest.approx(1.0) and ap[2] is None


def test_false_positive_ranked_above_true_lowers_ap():
    preds = {"a.jpg": [[0, 0.9, 300, 300, 340, 340], [0, 0.8, 10, 10, 50, 50]],
             "b.jpg": [[0, 0.7, 100, 100, 150, 150]]}
    ap = map50(GT, preds, [0])[0]
    assert 0.5 < ap < 1.0


def test_duplicate_detection_counts_once():
    preds = {"a.jpg": [[0, 0.9, 10, 10, 50, 50], [0, 0.8, 11, 11, 50, 50]]}
    only_a = {"a.jpg": GT["a.jpg"]}
    assert map50(only_a, preds, [0])[0] == pytest.approx(1.0)  # второй — FP, но ниже по рангу
    preds_low_first = {"a.jpg": [[0, 0.5, 300, 300, 310, 310], [0, 0.9, 10, 10, 50, 50]]}
    assert map50(only_a, preds_low_first, [0])[0] == pytest.approx(1.0)


def test_missed_objects():
    assert map50(GT, {}, [0])[0] == 0.0


def test_frame_level_counts():
    preds = {"a.jpg": [[0, 0.6, 0, 0, 1, 1]], "c.jpg": [[1, 0.9, 0, 0, 1, 1]]}
    r = frame_level(GT, preds, problems={0, 1}, thr=0.5)
    assert r["positives"] == 2 and r["negatives"] == 1
    assert r["recall"] == 0.5 and r["false_alarm"] == 1.0
    r = frame_level(GT, preds, problems={0}, thr=0.5)
    assert r["false_alarm"] == 0.0   # класс 1 не считается проблемой


def test_per_class_frame_recall():
    preds = {"a.jpg": [[0, 0.6, 0, 0, 1, 1]], "c.jpg": [[1, 0.9, 0, 0, 1, 1]]}
    s = per_class_frame_recall(GT, preds, thr=0.5)
    assert s[0] == (2, 1, 0) and s[1] == (1, 0, 1)


def test_pick_threshold_respects_false_alarm_budget():
    import sys
    from ddet.config import ROOT
    sys.path.insert(0, str(ROOT / "tools"))
    from calibrate import pick_threshold
    truth = {f"p{i}": [[0, 0, 0, 1, 1]] for i in range(10)} | {f"n{i}": [] for i in range(10)}
    preds = {f"p{i}": [[0, 0.3 + 0.05 * i, 0, 0, 1, 1]] for i in range(10)}
    preds |= {"n0": [[0, 0.6, 0, 0, 1, 1]]}                  # один чистый кадр с уверенностью 0.6
    thr, r = pick_threshold(truth, preds, {0}, max_fa=0.0)
    assert thr > 0.6 and r["false_alarm"] == 0.0
    thr2, r2 = pick_threshold(truth, preds, {0}, max_fa=0.1)
    assert thr2 < thr and r2["recall"] > r["recall"]


def test_recommended_conf_from_sidecar(tmp_path):
    import json
    from ddet.modelinfo import recommended_conf
    w = tmp_path / "m.pth"
    assert recommended_conf(w, 0.4) == 0.4
    w.with_suffix(".json").write_text(json.dumps({"defect_conf": 0.27}), encoding="utf-8")
    assert recommended_conf(w, 0.4) == 0.27
