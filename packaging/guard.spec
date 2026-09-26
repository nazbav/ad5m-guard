# PyInstaller: AD5M Guard — папка с exe (onedir: быстрый старт, антивирусы спокойнее, чем к onefile).
# Сборка: python packaging/build.py  (модели и демо-кадры должны лежать в build/models и build/demo)
from pathlib import Path

import os

ROOT = Path(SPECPATH).parent
BUILD = ROOT / "build"
# Только модели из models.yaml (список готовит build.py), остальное в build/models в exe не едет.
MODELS = [f for f in os.environ.get("GUARD_MODEL_FILES", "defects.onnx;scene.onnx").split(";") if f]

a = Analysis(
    [str(ROOT / "guard_entry.py")],
    pathex=[str(ROOT)],
    datas=[
        (str(ROOT / "guard" / "web" / "static"), "guard/web/static"),
        *[(str(BUILD / "models" / f), "models") for f in MODELS],
        (str(BUILD / "demo"), "demo"),
    ],
    hiddenimports=["emulator.fleet", "emulator.servers", "emulator.virtual", "pystray._win32"],
    excludes=["torch", "torchvision", "ultralytics", "matplotlib", "tkinter", "pytest", "scipy", "pandas", "IPython"],
    noarchive=True,        # код — отдельными .pyc: build.py --patch обновляет его без PyInstaller
)
# Видео приложение не читает (только JPEG-снимки) — ffmpeg из OpenCV (~30 МБ) не нужен.
a.binaries = [b for b in a.binaries if "opencv_videoio_ffmpeg" not in b[0]]
a.datas = [d for d in a.datas if "opencv_videoio_ffmpeg" not in d[0]]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="AD5M-Guard",
    icon=str(BUILD / "guard.ico"),
    console=False,
    version=str(BUILD / "version.txt"),
)
coll = COLLECT(exe, a.binaries, a.datas, name="AD5M-Guard")
