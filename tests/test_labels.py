import pytest

from ddet.labels import Box, iou, merge_copies, parse_line, parse_text, remap


def test_bbox_passes_through():
    b = parse_line("3 0.5 0.4 0.2 0.1")
    assert b == Box(3, pytest.approx(0.5), pytest.approx(0.4), pytest.approx(0.2), pytest.approx(0.1))


def test_polygon_becomes_enclosing_box():
    # треугольник (0.1,0.2) (0.5,0.8) (0.3,0.4) → рамка x 0.1..0.5, y 0.2..0.8
    b = parse_line("7 0.1 0.2 0.5 0.8 0.3 0.4")
    assert b.cls == 7
    assert (b.xc, b.yc, b.w, b.h) == pytest.approx((0.3, 0.5, 0.4, 0.6))


def test_polygon_is_not_read_as_first_four_numbers():
    # Старый train.py брал первые 4 координаты полигона как xc yc w h.
    b = parse_line("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9")
    assert (b.xc, b.yc, b.w, b.h) == pytest.approx((0.5, 0.5, 0.8, 0.8))


def test_box_clipped_to_frame():
    b = parse_line("1 0.95 0.5 0.2 0.2")  # правый край вылезает за 1.0
    assert b.xc + b.w / 2 == pytest.approx(1.0)
    assert b.w == pytest.approx(0.15)


def test_degenerate_box_dropped():
    assert parse_line("1 0.5 0.5 0.0 0.3") is None
    assert parse_line("1 0.2 0.2 0.2 0.2 0.2 0.2") is None  # полигон из одной точки


@pytest.mark.parametrize("line", ["1 0.5 0.5", "1 0.1 0.1 0.2 0.2 0.3", "x 0.1 0.1 0.2 0.2"])
def test_malformed_line_raises(line):
    with pytest.raises(ValueError):
        parse_line(line)


def test_parse_text_skips_blank_lines():
    assert len(parse_text("0 0.5 0.5 0.1 0.1\n\n  \n1 0.5 0.5 0.2 0.2\n")) == 2


def test_remap_merges_and_drops():
    src = ["PEI_PLATE", "__ER_WEB", "__ER_STRINGING"]
    mapping = {"PEI_PLATE": None, "__ER_WEB": "stringing", "__ER_STRINGING": "stringing"}
    boxes = [Box(0, .5, .5, .1, .1), Box(1, .5, .5, .1, .1), Box(2, .5, .5, .1, .1)]
    out = remap(boxes, src, mapping, ["spaghetti", "stringing"])
    assert [b.cls for b in out] == [1, 1]


def test_remap_same_name_passes_without_mapping():
    out = remap([Box(0, .5, .5, .1, .1)], ["stringing"], {}, ["spaghetti", "stringing"])
    assert [b.cls for b in out] == [1]


def test_configs_cover_each_others_classes():
    from ddet.config import CONFIG_DIR, load_classes
    defects, scene = load_classes(CONFIG_DIR / "defects.yaml"), load_classes(CONFIG_DIR / "scene.yaml")
    assert not set(defects.names) & set(scene.names)
    for a, b in ((defects, scene), (scene, defects)):
        for n in b.names:
            assert a.roboflow_map.get(n, "missing") is None, f"{a.name}: класс {n} должен отбрасываться"


def test_remap_unknown_class_is_an_error():
    with pytest.raises(KeyError):
        remap([Box(0, .5, .5, .1, .1)], ["NEW"], {}, ["a"])


def test_iou():
    a = Box(0, 0.5, 0.5, 0.2, 0.2)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, Box(0, 0.9, 0.9, 0.1, 0.1)) == 0.0
    assert iou(a, Box(0, 0.55, 0.5, 0.2, 0.2)) == pytest.approx(0.15 / 0.25)


def test_merge_copies_collapses_same_box_and_keeps_extra():
    a = Box(1, 0.5, 0.5, 0.2, 0.2)
    a_shifted = Box(1, 0.51, 0.5, 0.2, 0.2)
    extra = Box(1, 0.1, 0.1, 0.05, 0.05)
    merged, conflict = merge_copies([[a], [a_shifted, extra]])
    assert merged == [a, extra]
    assert not conflict


def test_merge_copies_detects_conflicting_classes():
    merged, conflict = merge_copies([[Box(3, 0.5, 0.5, 0.2, 0.2)], [Box(9, 0.5, 0.5, 0.21, 0.2)]])
    assert conflict


def test_merge_copies_different_classes_in_different_places_is_fine():
    _, conflict = merge_copies([[Box(3, 0.2, 0.2, 0.1, 0.1)], [Box(9, 0.8, 0.8, 0.1, 0.1)]])
    assert not conflict
