import json
import sys

import cv2
import numpy as np
import pytest

from ddet.config import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import review  # noqa: E402

DEFECTS = {"spaghetti", "stringing", "warping", "cracks", "garbage", "detached"}
SHEETS = {"0001": ["printer_20250101_100000.jpg", "printer_20250101_100030.jpg"]}


def test_parse_all_token_kinds():
    text = ("# 0001\n"
            "1 a c=garbage +spaghetti@10,20 +garbage@1,2,30,40 glass misplaced // комментарий\n"
            "2 . nocam\n")
    d = review.parse_decisions(text, SHEETS, DEFECTS)
    first = d["printer_20250101_100000.jpg"]
    assert first.keep == {"a": None, "c": "garbage"}
    assert first.add == [("spaghetti", [10.0, 20.0]), ("garbage", [1.0, 2.0, 30.0, 40.0])]
    assert first.flags == {"glass", "misplaced"}
    second = d["printer_20250101_100030.jpg"]
    assert second.clean and second.flags == {"nocam"}


@pytest.mark.parametrize("text", [
    "1 a\n",                         # нет заголовка листа
    "# 0002\n1 a\n",                 # нет такого листа
    "# 0001\n3 a\n",                 # нет такого кадра
    "# 0001\n1 +plane@1,2\n",        # неизвестный класс
    "# 0001\n1 a=plane\n",           # неизвестный класс
    "# 0001\n1 ???\n",               # мусорный токен
])
def test_parse_rejects_bad_input(text):
    with pytest.raises(ValueError):
        review.parse_decisions(text, SHEETS, DEFECTS)


def test_dedupe_lines_keeps_first_of_same_class_only():
    lines = ["10 0.5 0.5 0.2 0.2", "10 0.51 0.5 0.2 0.2", "0 0.5 0.5 0.2 0.2"]
    assert review.dedupe_lines(lines, 640, 480) == ["10 0.5 0.5 0.2 0.2", "0 0.5 0.5 0.2 0.2"]


@pytest.fixture
def batch(tmp_path):
    b = tmp_path / "batch"
    (b / "images").mkdir(parents=True)
    for n in SHEETS["0001"]:
        cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))[1].tofile(str(b / "images" / n))
    (b / "review" / "decisions").mkdir(parents=True)
    cands = {
        SHEETS["0001"][0]: {"size": [640, 480],
                            "defects": [{"id": "a", "cls": "spaghetti", "conf": 0.8, "box": [100, 100, 200, 200]},
                                        {"id": "b", "cls": "garbage", "conf": 0.2, "box": [300, 300, 310, 310]}],
                            "scene": [{"cls": "pei_plate", "conf": 0.95, "box": [0, 200, 640, 480]},
                                      {"cls": "printer_head", "conf": 0.3, "box": [0, 0, 100, 100]},
                                      {"cls": "pei_misplaced", "conf": 0.6, "box": [0, 400, 80, 480]}]},
        SHEETS["0001"][1]: {"size": [640, 480], "defects": [],
                            "scene": [{"cls": "pei_plate", "conf": 0.9, "box": [0, 200, 640, 480]}]},
    }
    (b / "review" / "candidates.json").write_text(json.dumps(cands), encoding="utf-8")
    (b / "review" / "sheets.json").write_text(json.dumps({"0001": {"session": 0, "frames": SHEETS["0001"]}}), encoding="utf-8")
    return b


def labels(out, frame):
    return (out / "labels" / frame.replace(".jpg", ".txt")).read_text(encoding="utf-8").split("\n")


def test_apply_builds_labels(batch, tmp_path):
    (batch / "review" / "decisions" / "x.txt").write_text(
        "# 0001\n1 a=detached misplaced +garbage@10,10,20,20\n2 . glass\n", encoding="utf-8")
    out = tmp_path / "labeled"
    review.cmd_apply(batch, out, "cpu", allow_partial=False)
    names = review.labeling_names()
    first = [l.split()[0] for l in labels(out, SHEETS["0001"][0]) if l]
    assert sorted(names[int(c)] for c in first) == ["detached", "garbage", "pei_misplaced", "pei_plate"]
    second = [names[int(l.split()[0])] for l in labels(out, SHEETS["0001"][1]) if l]
    assert second == ["glass_plate"]  # стекло вместо PEI; голова ниже порога не попала


def test_apply_refuses_incomplete_sheet(batch, tmp_path):
    (batch / "review" / "decisions" / "x.txt").write_text("# 0001\n1 .\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        review.cmd_apply(batch, tmp_path / "labeled", "cpu", allow_partial=True)


def test_check_catches_missing_letter(batch):
    f = batch / "review" / "decisions" / "x.txt"
    f.write_text("# 0001\n1 z\n2 .\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="нет кандидатов"):
        review.cmd_check(batch, f)
