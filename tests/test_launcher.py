import importlib.machinery
import importlib.util
import os
import subprocess
import sys

import pytest

from ddet.config import ROOT
from ddet.report import md_to_html
from update_data import latest_versions


def load_launcher():
    loader = importlib.machinery.SourceFileLoader("launcher", str(ROOT / "launcher.pyw"))
    spec = importlib.util.spec_from_loader("launcher", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def test_command_goes_through_runner():
    launcher = load_launcher()
    cmd = launcher.command("run_video.py", "--all", 10, open_after=ROOT / "x.html")
    assert cmd[1:4] == ["-m", "ddet.runner", "--open"]
    assert cmd[-3:] == ["run_video.py", "--all", "10"]


def test_next_batch_skips_existing(tmp_path):
    launcher = load_launcher()
    (tmp_path / "batch1").mkdir()
    (tmp_path / "batch2.zip").write_text("")
    assert launcher.next_batch(tmp_path).name == "batch3"


def test_window_builds():
    launcher = load_launcher()
    tk = launcher.tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("нет дисплея")
    root.withdraw()
    try:
        launcher.App(root)
        root.update_idletasks()
    finally:
        root.destroy()


def test_runner_reports_failure_and_success(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text("import sys; print('arg', sys.argv[1])", encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("raise SystemExit('плохо')", encoding="utf-8")
    env = dict(os.environ, DDET_NO_PAUSE="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-m", "ddet.runner", str(ok), "x"], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0 and "arg x" in r.stdout
    r = subprocess.run([sys.executable, "-m", "ddet.runner", str(bad)], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 1 and "плохо" in r.stdout


def test_latest_version_per_project(tmp_path):
    for name in ["proj_v3_yolov11.zip", "proj_v17_yolov11.zip", "other_v2_yolov11.zip", "junk.zip"]:
        (tmp_path / name).write_text("")
    assert [p.name for p in latest_versions(tmp_path)] == ["other_v2_yolov11.zip", "proj_v17_yolov11.zip"]


def test_md_to_html_tables_lists_links():
    html = md_to_html("# Заголовок\n\n| a | b |\n|---|---:|\n| x | 12 |\n\n- **жирно** [ссылка](s.html)\n\n<script>")
    assert "<h1>Заголовок</h1>" in html
    assert "<th>a</th>" in html and "<td class=num>12</td>" in html and "---" not in html
    assert "<li><b>жирно</b> <a href=\"s.html\">ссылка</a></li>" in html
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_label_tool_draw_delete_and_save(tmp_path):
    import cv2
    import numpy as np
    import yaml
    import label_tool
    folder = tmp_path / "set"
    (folder / "images").mkdir(parents=True)
    (folder / "labels").mkdir()
    cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))[1].tofile(str(folder / "images" / "printer_20250101_100000.jpg"))
    (folder / "labels" / "printer_20250101_100000.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (folder / "data.yaml").write_text(yaml.safe_dump({"names": ["spaghetti", "garbage", "pei_plate"]}), encoding="utf-8")
    try:
        root = label_tool.tk.Tk()
    except label_tool.tk.TclError:
        pytest.skip("нет дисплея")
    root.withdraw()
    try:
        app = label_tool.Labeler(root, folder)
        s = app.SCALE

        class Ev:
            def __init__(self, x, y):
                self.x, self.y = x * s, y * s

        app.current.set(1)                                   # garbage
        app.on_press(Ev(10, 10)); app.on_drag(Ev(60, 50)); app.on_release(Ev(60, 50))
        app.on_press(Ev(320, 240))                           # выделить старую рамку spaghetti
        app.delete()
        app.save()
        lines = (folder / "labels" / "printer_20250101_100000.txt").read_text(encoding="utf-8").split()
        assert lines[0] == "1" and len(lines) == 5          # осталась только новая рамка garbage
    finally:
        root.destroy()
