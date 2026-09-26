from evaluate import problem_classes


def test_defect_model_all_classes_are_problems():
    names = {0: "spaghetti", 1: "stringing", 2: "detached"}
    assert problem_classes(names) == {0, 1, 2}


def test_scene_model_plate_and_head_are_normal():
    names = {0: "pei_plate", 1: "printer_head", 2: "pei_misplaced", 3: "test_line", 4: "hand", 5: "head_no_cover"}
    assert problem_classes(names) == {2, 5}
