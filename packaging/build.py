"""Сборка AD5M Guard в exe: тесты → PyInstaller → проверка готового exe на демо-принтерах → zip.

    .venv-build\\Scripts\\python packaging\\build.py

Перед сборкой (в окружении с Ultralytics, один раз на новую модель):
    .venv\\Scripts\\python tools\\export_onnx.py      → build/models/*.onnx (+ сверка с исходником)
    .venv\\Scripts\\python tools\\make_demo.py        → build/demo (кадры демо-режима)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DIST = Path(os.environ.get("GUARD_DIST", ROOT / "dist"))        # другая папка, если прежняя сборка сейчас запущена
sys.path.insert(0, str(ROOT))
from guard import __version__  # noqa: E402


def step(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def icon() -> None:
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, 248, 248), 56, fill=(37, 99, 235))
    d.polygon([(128, 48), (200, 84), (200, 136), (128, 208), (56, 136), (56, 84)], fill=(255, 255, 255))
    d.polygon([(128, 92), (164, 108), (164, 132), (128, 168), (92, 132), (92, 108)], fill=(37, 99, 235))
    img.save(BUILD / "guard.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def version_file() -> None:
    nums = [int(x) for x in __version__.split(".")] + [0]
    ver = ", ".join(map(str, nums[:4]))
    (BUILD / "version.txt").write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers=({ver}), prodvers=({ver})),
  kids=[StringFileInfo([StringTable('041904B0', [
    StringStruct('CompanyName', 'AD5M Guard'), StringStruct('FileDescription', 'AD5M Guard — контроль печати FlashForge AD5M'),
    StringStruct('FileVersion', '{__version__}'), StringStruct('ProductName', 'AD5M Guard'),
    StringStruct('ProductVersion', '{__version__}'), StringStruct('OriginalFilename', 'AD5M-Guard.exe')])]),
    VarFileInfo([VarStruct('Translation', [1049, 1200])])])""", encoding="utf-8")


def smoke(exe: Path, wait_stop: bool = True) -> None:
    """Запустить готовый exe в демо-режиме и убедиться, что он видит принтеры и распознаёт."""
    port = 18765
    data = Path(tempfile.mkdtemp(prefix="guard-smoke-"))
    env = {**os.environ, "GUARD_DEMO_FAST": "1"}           # в демо сбой на AD5M-01 начинается почти сразу
    proc = subprocess.Popen([str(exe), "--demo", "--no-tray", "--no-browser", "--data-dir", str(data), "--port", str(port)],
                            env=env)
    try:
        deadline, s = time.time() + 120, None
        while time.time() < deadline:
            try:
                s = requests.get(f"http://127.0.0.1:{port}/api/state", timeout=2).json()
                if s["printers"] and all(p["state"]["status"] not in ("unknown", "offline") for p in s["printers"]) \
                        and all(p["has_frame"] for p in s["printers"]) and s.get("detector_state") != "loading":
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise SystemExit(f"exe не поднялся или не видит демо-принтеры: {s}")
        assert s["detector"], "в exe не загрузились модели"
        assert requests.get(f"http://127.0.0.1:{port}/", timeout=5).ok
        pid = s["printers"][0]["id"]
        assert requests.get(f"http://127.0.0.1:{port}/api/frame/{pid}.jpg", timeout=5).content[:2] == b"\xff\xd8"
        print("exe работает:", ", ".join(f"{p['name']}={p['state']['status']}" for p in s["printers"]), flush=True)
        if not wait_stop:
            return
        # Демо AD5M-01: чистая печать, затем настоящий сбой (ком нити) — exe должен остановить её сам,
        # а чистые AD5M-02/03 — не тронуть.
        deadline = time.time() + 480
        while time.time() < deadline:
            ev = requests.get(f"http://127.0.0.1:{port}/api/events?limit=200", timeout=5).json()
            ev = ev if isinstance(ev, list) else ev.get("events", [])
            stopped = [e for e in ev if e["kind"] == "stopped"]
            if stopped:
                assert {e["printer"] for e in stopped} == {"AD5M-01"}, f"остановлен не тот принтер: {stopped}"
                print(f"exe остановил печать со сбоем: {stopped[0]['printer']} — {stopped[0]['detail']}")
                break
            time.sleep(5)
        else:
            raise SystemExit("exe не остановил демо-печать со сбоем за 8 минут")
    finally:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(data, ignore_errors=True)


# Только для сборки — в exe не попадают, их лицензии в список не нужны.
BUILD_ONLY = {"pyinstaller", "pyinstaller-hooks-contrib", "altgraph", "pefile", "pywin32-ctypes", "pip", "setuptools",
              "wheel", "pytest", "pluggy", "iniconfig", "packaging", "colorama", "pygments"}


def third_party_licenses(dst: Path) -> None:
    """Лицензии библиотек, которые едут в exe (всё окружение сборки, кроме инструментов сборки)."""
    from importlib.metadata import distributions
    parts, rows = [], []
    for dist in sorted(distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        if name.lower() in BUILD_ONLY:
            continue
        lic = dist.metadata.get("License-Expression") or next(
            (c.split("::")[-1].strip() for c in dist.metadata.get_all("Classifier") or [] if c.startswith("License ::")),
            None) or (dist.metadata.get("License") or "").splitlines()[0:1] or ["см. текст ниже"]
        lic = lic if isinstance(lic, str) else lic[0]
        rows.append(f"{name} {dist.version} — {lic}")
        texts = [f for f in dist.files or [] if "licen" in f.name.lower() or "copying" in f.name.lower() or "notice" in f.name.lower()]
        for f in texts:
            try:
                parts.append(f"\n{'=' * 78}\n{name} {dist.version}: {f.name}\n{'=' * 78}\n{f.locate().read_text(encoding='utf-8', errors='replace')}")
            except OSError:
                pass
    head = ("AD5M Guard распространяется под GNU AGPL-3.0 (файл LICENSE).\n"
            "В сборку входят Python (PSF License) и библиотеки ниже; модели обучены с Ultralytics YOLO (AGPL-3.0).\n\n")
    dst.write_text(head + "\n".join(rows) + "\n" + "".join(parts), encoding="utf-8")


def manifest_models() -> list[str]:
    """Какие файлы из build/models нужны exe (models.yaml), чтобы не тащить лишние модели в сборку."""
    import yaml
    m = {"defects": "defects.onnx", "scene": "scene.onnx", "second": None}
    path = BUILD / "models" / "models.yaml"
    if path.exists():
        m |= yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    files = [m["defects"], m["scene"]] + ([m["second"]] if m["second"] else []) + (["models.yaml"] if path.exists() else [])
    missing = [f for f in files if not (BUILD / "models" / f).exists()]
    if missing:
        raise SystemExit(f"в build/models нет {missing}")
    return files


def remove_app_dir(app: Path) -> None:
    """Старую сборку — целиком или никак: если её exe запущен, файлы заняты, и удаление «с пропуском ошибок»
    оставило бы наполовину удалённую программу у того, кто ею пользуется."""
    if not app.exists():
        return
    old = app.with_name(app.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    try:
        app.rename(old)                  # на Windows не получится, пока что-то из папки запущено
    except OSError as e:
        raise SystemExit(f"{app} занята (запущен AD5M-Guard.exe?) — закройте его или соберите в другую папку: "
                         f"set GUARD_DIST=...  ({e})")
    shutil.rmtree(old)


def docs(app: Path) -> None:
    shutil.copy2(ROOT / "docs" / "USER_GUIDE.md", app / "ИНСТРУКЦИЯ.md")
    shutil.copy2(ROOT / "CHANGELOG.md", app / "ЧТО НОВОГО.md")
    shutil.copy2(ROOT / "LICENSE", app / "LICENSE.txt")


def patch(app: Path) -> None:
    """Только код и данные — без PyInstaller: .py → .pyc прямо в _internal (сборка с noarchive), статика,
    модели, демо-кадры. Секунды вместо минут, если менялись только guard/ и emulator/."""
    import py_compile
    internal = app / "_internal"
    if not (internal / "guard").is_dir():
        raise SystemExit(f"{app}: нет готовой сборки с noarchive — сначала обычная сборка")
    for pkg in ("guard", "emulator"):
        for src in (ROOT / pkg).rglob("*.py"):
            rel = src.relative_to(ROOT).with_suffix(".pyc")
            py_compile.compile(str(src), cfile=str(internal / rel), doraise=True)
    shutil.copytree(ROOT / "guard" / "web" / "static", internal / "guard" / "web" / "static", dirs_exist_ok=True)
    for f in manifest_models():
        shutil.copy2(BUILD / "models" / f, internal / "models" / f)
    shutil.copytree(BUILD / "demo", internal / "demo", dirs_exist_ok=True)
    docs(app)


def main() -> None:
    """Режимы: (по умолчанию) релиз — тесты параллельно со сборкой, проверка остановки сбоя, zip;
    --fast — сборка exe без тестов и zip, проверка «запустился»; --patch — без PyInstaller (секунды)."""
    fast, patch_only = "--fast" in sys.argv, "--patch" in sys.argv
    for need in (BUILD / "models" / "defects.onnx", BUILD / "demo"):
        if not need.exists():
            raise SystemExit(f"нет {need} — см. docstring: export_onnx.py и make_demo.py")
    os.environ["GUARD_MODEL_FILES"] = ";".join(manifest_models())
    app = DIST / "AD5M-Guard"
    # Неопределённые имена (забытый import) тесты ловят не всегда — pyflakes ловит всегда, за секунду.
    lint = subprocess.run([sys.executable, "-m", "pyflakes", "guard", "emulator", "guard_entry.py"], cwd=ROOT,
                          capture_output=True, text=True)
    errors = [l for l in lint.stdout.splitlines() if "undefined name" in l or "syntax" in l.lower()]
    if errors:
        raise SystemExit("ошибки в коде:\n" + "\n".join(errors))
    if patch_only:
        step("обновление кода в готовой сборке")
        patch(app)
        smoke(app / "AD5M-Guard.exe", wait_stop=False)
        return
    tests = None if fast else subprocess.Popen([sys.executable, "-m", "pytest", "-q", "tests/product"], cwd=ROOT)
    step("PyInstaller" + ("" if fast else " (тесты идут параллельно)"))
    icon()
    version_file()
    remove_app_dir(app)
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(DIST),
                    "--workpath", str(BUILD / "pyinstaller"), str(ROOT / "packaging" / "guard.spec")], cwd=ROOT, check=True)
    docs(app)
    if tests is not None and tests.wait() != 0:
        raise SystemExit("тесты не прошли")
    step("проверка готового exe")
    smoke(app / "AD5M-Guard.exe", wait_stop=not fast)
    if fast:
        print(f"\nГотово (быстро, без zip): {app}")
        return
    third_party_licenses(app / "THIRD_PARTY_LICENSES.txt")
    step("zip")
    archive = DIST / f"AD5M-Guard-{__version__}-win64.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:   # модели/DLL почти не жмутся
        for p in app.rglob("*"):
            z.write(p, Path("AD5M-Guard") / p.relative_to(app))
    import hashlib
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (DIST / f"{archive.name}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    size = sum(p.stat().st_size for p in app.rglob("*") if p.is_file()) / 2**20
    print(f"\nГотово: {archive} ({archive.stat().st_size / 2**20:.0f} МБ zip, {size:.0f} МБ в папке)\nSHA-256: {digest}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    main()
