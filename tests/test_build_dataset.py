import csv
import io
import zipfile
from collections import defaultdict

import numpy as np
import pytest
import yaml
from PIL import Image

import build_dataset
from ddet.labels import parse_text

SRC_NAMES = ["PEI_PLATE", "PRINTER_HEAD", "TEST_FILAMENT_LINE", "__ER_CRACKS", "__ER_GARBAGE", "__ER_PEI_SET",
             "__ER_PEOPLE_HAND", "__ER_SPAGHETTI", "__ER_STRINGING", "__ER_WARPING", "__ER_WEB"]


def _jpeg(seed):
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def roboflow_zip(tmp_path):
    path = tmp_path / "export.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data.yaml", yaml.safe_dump({"nc": len(SRC_NAMES), "names": SRC_NAMES}))
        n = 0
        # 6 печатей по 5 кадров, между печатями по 3 часа
        for day in range(6):
            for k in range(5):
                name = f"printer_202501{10 + day:02d}_12{k:02d}00_jpg.rf.{n:032x}.jpg"
                z.writestr(f"train/images/{name}", _jpeg(n))
                # спагетти полигоном + стол, который должен выброситься
                z.writestr(f"train/labels/{name[:-4]}.txt",
                           "7 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n0 0.5 0.5 0.9 0.9\n")
                n += 1
        # 40 картинок из интернета, у каждой вторая — струны (WEB сливается в stringing)
        for k in range(40):
            name = f"photo_{k}_jpg.rf.{n:032x}.jpg"
            z.writestr(f"valid/images/{name}", _jpeg(n))
            z.writestr(f"valid/labels/{name[:-4]}.txt", "10 0.5 0.5 0.2 0.2\n" if k % 2 else "8 0.3 0.3 0.1 0.1\n")
            n += 1
        # точный дубль
        z.writestr("test/images/dup_jpg.rf.ffff.jpg", _jpeg(n - 1))
        z.writestr("test/labels/dup_jpg.rf.ffff.txt", "8 0.3 0.3 0.1 0.1\n")
    return path


def test_build(tmp_path, roboflow_zip):
    out = tmp_path / "ds"
    build_dataset.main([str(roboflow_zip), "--out", str(out)])

    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    assert "path" not in data and data["train"] == "train/images"
    assert names[0] == "spaghetti" and data["nc"] == len(names)

    images = {s: sorted(p.name for p in (out / s / "images").iterdir()) for s in ("train", "val", "test")}
    assert sum(map(len, images.values())) == 30 + 40  # дубль убран
    for s, files in images.items():
        for f in files:
            assert (out / s / "labels" / (f.rsplit(".", 1)[0] + ".txt")).exists()

    stringing = names_to_id(names)["stringing"]
    spaghetti = names_to_id(names)["spaghetti"]
    own_label = next((out / "train" / "labels").glob("own_*.txt"), None) or next(out.glob("*/labels/own_*.txt"))
    boxes = parse_text(own_label.read_text(encoding="utf-8"))
    assert [b.cls for b in boxes] == [spaghetti]  # стол выброшен
    assert (boxes[0].xc, boxes[0].w) == pytest.approx((0.3, 0.4))  # полигон → рамка

    all_classes = {b.cls for p in out.glob("*/labels/ext_*.txt") for b in parse_text(p.read_text(encoding="utf-8"))}
    assert all_classes == {stringing}  # WEB и STRINGING слились

    with open(out / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split_of_group = defaultdict(set)
    for r in rows:
        split_of_group[r["group"]].add(r["split"])
    assert all(len(v) == 1 for v in split_of_group.values()), "группа разнесена по выборкам"
    assert len({r["group"] for r in rows if r["domain"] == "own"}) == 6  # по группе на печать

    own_test = (out / "test_own.txt").read_text(encoding="utf-8").split()
    assert all(line.startswith("./test/images/own_") for line in own_test)


def test_rebuild_replaces_own_output(tmp_path, roboflow_zip):
    out = tmp_path / "ds"
    build_dataset.main([str(roboflow_zip), "--out", str(out)])
    (out / "train" / "images" / "stale.jpg").write_bytes(b"x")
    build_dataset.main([str(roboflow_zip), "--out", str(out)])
    assert not (out / "train" / "images" / "stale.jpg").exists()


def test_refuses_foreign_dir(tmp_path, roboflow_zip):
    out = tmp_path / "important"
    out.mkdir()
    (out / "keep.txt").write_text("data")
    with pytest.raises(SystemExit):
        build_dataset.main([str(roboflow_zip), "--out", str(out)])
    assert (out / "keep.txt").exists()


def names_to_id(names):
    return {n: i for i, n in names.items()}


def test_scene_config_keeps_only_own_frames(tmp_path, roboflow_zip):
    from ddet.config import CONFIG_DIR
    out = tmp_path / "scene"
    build_dataset.main([str(roboflow_zip), "--config", str(CONFIG_DIR / "scene.yaml"), "--out", str(out)])
    names = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))["names"]
    files = [p.name for p in out.glob("*/images/*")]
    assert len(files) == 30 and all(f.startswith("own_") for f in files)
    classes = {names[b.cls] for p in out.glob("*/labels/*.txt") for b in parse_text(p.read_text(encoding="utf-8"))}
    assert classes == {"pei_plate"}  # спагетти в модели сцены не нужно


def test_glob_sources(tmp_path, roboflow_zip, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "ds"
    build_dataset.main(["*.zip", "--out", str(out)])
    assert len(list(out.glob("*/images/*"))) == 70


def test_own_repeat_and_val_own(tmp_path, roboflow_zip):
    out = tmp_path / "ds"
    build_dataset.main([str(roboflow_zip), "--out", str(out), "--own-repeat", "3", "--val-own"])
    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data["train"] == "train.txt" and data["val"] == "val_own.txt" and data["test"] == "test/images"
    train_files = [p.name for p in (out / "train" / "images").iterdir()]
    own, ext = sum(f.startswith("own_") for f in train_files), sum(f.startswith("ext_") for f in train_files)
    lines = (out / "train.txt").read_text(encoding="utf-8").split()
    assert len(lines) == ext + 3 * own
    assert all((out / l).exists() for l in lines)

    from evaluate import split_images
    val = split_images(out / "data.yaml", "val")
    assert val and all(p.name.startswith("own_") for p in val)


def test_rebuild_with_more_data_keeps_previous_splits(tmp_path, roboflow_zip):
    out = tmp_path / "ds"
    build_dataset.main([str(roboflow_zip), "--out", str(out)])
    before = {r["file"]: r["split"] for r in csv.DictReader(open(out / "manifest.csv", encoding="utf-8"))}
    extra = tmp_path / "extra.zip"
    with zipfile.ZipFile(extra, "w") as z:
        z.writestr("data.yaml", yaml.safe_dump({"nc": len(SRC_NAMES), "names": SRC_NAMES}))
        for k in range(30):
            name = f"new_{k}_jpg.rf.{k:032x}.jpg"
            z.writestr(f"train/images/{name}", _jpeg(1000 + k))
            z.writestr(f"train/labels/{name[:-4]}.txt", "8 0.5 0.5 0.1 0.1\n")
    build_dataset.main([str(roboflow_zip), str(extra), "--out", str(out)])
    after = {r["file"]: r["split"] for r in csv.DictReader(open(out / "manifest.csv", encoding="utf-8"))}
    assert all(after[f] == s for f, s in before.items())
    assert len(after) == len(before) + 30


def test_labeled_set_in_new_class_names(tmp_path, roboflow_zip):
    """Своя разметка (data/labeled) — в классах обеих моделей: свои переходят сами в себя, чужие отбрасываются."""
    import numpy as np
    from ddet.config import labeling_names
    names = labeling_names()
    lab = tmp_path / "labeled"
    (lab / "images").mkdir(parents=True)
    (lab / "labels").mkdir()
    import cv2
    for k in range(4):
        n = f"printer_20250601_12{k:02d}00"
        cv2.imencode(".jpg", np.random.default_rng(k).integers(0, 255, (48, 64, 3), dtype=np.uint8))[1].tofile(
            str(lab / "images" / f"{n}.jpg"))
        (lab / "labels" / f"{n}.txt").write_text(
            f"{names.index('spaghetti')} 0.5 0.5 0.2 0.2\n{names.index('pei_plate')} 0.5 0.7 0.9 0.5\n", encoding="utf-8")
    (lab / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
    out = tmp_path / "ds"
    build_dataset.main([str(roboflow_zip), str(lab), "--out", str(out)])
    ds_names = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))["names"]
    new = [p for p in out.glob("*/labels/own_20250601_*.txt")]
    assert len(new) == 4
    assert all([ds_names[b.cls] for b in parse_text(p.read_text(encoding="utf-8"))] == ["spaghetti"] for p in new)


def test_holdout_sessions_go_to_test_whole(tmp_path, roboflow_zip):
    hold = tmp_path / "holdout.yaml"
    hold.write_text(yaml.safe_dump({"ranges": [{"from": "2025-01-12 11:00", "to": "2025-01-12 13:00"},
                                               {"from": "2025-01-14 11:00", "to": "2025-01-14 13:00"}]}), encoding="utf-8")
    out = tmp_path / "ds"
    build_dataset.main([str(roboflow_zip), "--out", str(out), "--holdout", str(hold)])
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    own = [r for r in rows if r["domain"] == "own"]
    for r in own:
        day = r["file"][4:12]
        if day in ("20250112", "20250114"):
            assert r["split"] == "test", r
        else:
            assert r["split"] in ("train", "val"), r


def test_synthetic_frames_only_in_train(tmp_path, roboflow_zip):
    import numpy as np
    import cv2
    from ddet.config import labeling_names
    names = labeling_names()
    syn = tmp_path / "synth"
    (syn / "images").mkdir(parents=True)
    (syn / "labels").mkdir()
    for k in range(12):
        n = f"synth_printer_2025061{k % 6}_120000_{k:05d}"
        cv2.imencode(".jpg", np.random.default_rng(100 + k).integers(0, 255, (48, 64, 3), dtype=np.uint8))[1].tofile(
            str(syn / "images" / f"{n}.jpg"))
        (syn / "labels" / f"{n}.txt").write_text(f"{names.index('spaghetti')} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (syn / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
    out = tmp_path / "ds"
    build_dataset.main([str(roboflow_zip), str(syn), "--out", str(out)])
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    synth = [r for r in rows if r["file"].startswith("synth_")]
    assert len(synth) == 12 and all(r["split"] == "train" for r in synth)
