"""Окно запуска детектора брака AD5M — двойной клик по start.bat или ярлыку на рабочем столе.

Каждая задача открывается в своём окне консоли (ddet/runner.py): там видно
ход работы, окно не закрывается само. Отчёты открываются в браузере.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_CAMERA = "http://192.168.0.108:8080/?action=stream"
EMU_CONFIG = ROOT / "runs" / "emulation" / "printers.yaml"
SERVICE_CONFIG = ROOT / "config" / "printers.yaml"


def command(script: str, *args, open_after: Path | None = None) -> list[str]:
    cmd = [str(PY), "-m", "ddet.runner"]
    if open_after:
        cmd += ["--open", str(open_after)]
    return cmd + [script, *map(str, args)]


def launch(script: str, *args, open_after: Path | None = None) -> None:
    flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.Popen(command(script, *args, open_after=open_after), cwd=ROOT, creationflags=flags, env=env)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def next_batch(root: Path) -> Path:
    n = 1
    while (root / f"batch{n}").exists() or (root / f"batch{n}.zip").exists():
        n += 1
    return root / f"batch{n}"


def describe_model(name: str) -> str:
    p = ROOT / "models" / f"{name}.pt"
    if not p.exists():
        return "нет — обучите"
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("обучена %d.%m.%Y %H:%M")


def latest(pattern: str) -> Path | None:
    files = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


class App(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, padding=12)
        master.title("Детектор брака AD5M")
        master.minsize(560, 0)
        self.grid(sticky="nsew")
        self.columnconfigure((0, 1), weight=1, uniform="col")

        status = ttk.LabelFrame(self, text="Модели", padding=8)
        status.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.status = ttk.Label(status, justify="left")
        self.status.pack(anchor="w")
        self.refresh()

        check = self.group("Проверить модель на записях", 1, 0)
        self.button(check, "Видео или zip с кадрами…", self.check_files)
        self.button(check, "Папка с кадрами…", self.check_folder)
        self.button(check, "Все записанные печати", self.check_all)
        self.button(check, "Живая камера принтера…", self.live_camera)
        self.button(check, "Последний прогон", lambda: self.open(latest("runs/video/*/index.html")))

        emu = self.group("Эмуляция и сервис", 1, 1)
        row = ttk.Frame(emu)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="принтеров").pack(side="left")
        self.printers = tk.IntVar(value=3)
        ttk.Spinbox(row, from_=1, to=50, width=4, textvariable=self.printers).pack(side="left", padx=4)
        ttk.Label(row, text="скорость ×").pack(side="left")
        self.speed = tk.IntVar(value=10)
        ttk.Spinbox(row, from_=1, to=200, width=4, textvariable=self.speed).pack(side="left", padx=4)
        self.button(emu, "Запустить эмуляцию + сервис", self.emulate)
        self.button(emu, "Сервис для настоящих принтеров", self.service)
        self.button(emu, "Панель управления (если сервис запущен)", lambda: webbrowser.open("http://127.0.0.1:8765/"))
        self.button(emu, "Журнал событий сервиса", lambda: self.open(latest("runs/monitor*/events.csv")))

        model = self.group("Модель", 2, 0)
        self.button(model, "Приёмочный тест (≈15 мин)", self.benchmark)
        self.button(model, "Последний приёмочный тест", lambda: self.open(latest("runs/benchmark/*/report.html")))
        self.button(model, "Отчёт о модели брака", lambda: self.report("defects"))
        self.button(model, "Отчёт о модели сцены", lambda: self.report("scene"))
        self.button(model, "Обучить модель брака (~2 ч)", lambda: self.train("defects", "yolo26s.pt"))
        self.button(model, "Обучить модель сцены (~20 мин)", lambda: self.train("scene", "yolo26n.pt"))

        data = self.group("Данные", 2, 1)
        self.button(data, "Разметка кадров", self.label_tool)
        self.button(data, "Отобрать кадры на разметку", self.prepare_labeling)
        self.button(data, "Обновить данные и датасеты", lambda: launch("update_data.py"))
        self.button(data, "Открыть папку проекта", lambda: self.open(ROOT))

        bottom = ttk.Frame(self)
        bottom.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(bottom, text="Тесты", command=self.tests).pack(side="left")
        ttk.Button(bottom, text="Справка", command=lambda: subprocess.Popen(["notepad", str(ROOT / "README.md")])).pack(side="left", padx=6)
        ttk.Button(bottom, text="Обновить", command=self.refresh).pack(side="right")

    # ---------------------------------------------------------------- вид
    def group(self, title: str, row: int, col: int) -> ttk.LabelFrame:
        f = ttk.LabelFrame(self, text=title, padding=8)
        f.grid(row=row, column=col, sticky="nsew", padx=(0, 8) if col == 0 else 0, pady=4)
        return f

    def button(self, parent, text: str, cmd) -> None:
        ttk.Button(parent, text=text, command=cmd).pack(fill="x", pady=2)

    def refresh(self) -> None:
        run = latest("runs/video/*/index.html")
        self.status.config(text=f"Брак: {describe_model('defects')}\nСцена: {describe_model('scene')}\n"
                                f"Последний прогон по записям: {run.parent.name if run else '—'}")

    def open(self, path: Path | None) -> None:
        if path and Path(path).exists():
            os.startfile(path)
        else:
            messagebox.showinfo("Нет файла", "Пока нечего открывать — сначала запустите соответствующую задачу.")

    # ---------------------------------------------------------------- проверка
    def need_model(self) -> bool:
        if not (ROOT / "models" / "defects.pt").exists():
            messagebox.showwarning("Нет модели", "Сначала обучите модель брака.")
            return False
        return True

    def run_video(self, *sources, every: int = 10) -> None:
        if self.need_model():
            launch("run_video.py", *sources, "--every", every, "--open", "--out", ROOT / "runs" / "video" / stamp())

    def check_files(self) -> None:
        files = filedialog.askopenfilenames(title="Записи печати", filetypes=[
            ("Видео и архивы кадров", "*.mp4 *.mkv *.avi *.mov *.webm *.zip"), ("Все файлы", "*.*")])
        if files:
            self.run_video(*files, every=0)

    def check_folder(self) -> None:
        folder = filedialog.askdirectory(title="Папка с кадрами printer_*.jpg")
        if folder:
            self.run_video(folder)

    def check_all(self) -> None:
        if messagebox.askyesno("Все печати", "Прогнать модели по всем записанным печатям AD5M?\nЗаймёт около получаса."):
            self.run_video("--all")

    def live_camera(self) -> None:
        url = simpledialog.askstring("Живая камера", "Адрес потока камеры:", initialvalue=DEFAULT_CAMERA, parent=self)
        if url and self.need_model():
            messagebox.showinfo("Живая камера", "Остановить — Ctrl+C в окне консоли, отчёт откроется сам.")
            self.run_video(url, every=2)

    # ---------------------------------------------------------------- эмуляция и сервис
    def emulate(self) -> None:
        if not self.need_model():
            return
        EMU_CONFIG.unlink(missing_ok=True)
        launch("emulate_printer.py", "--all", "--printers", self.printers.get(), "--speed", self.speed.get())
        self.after(2000, self._start_monitor_when_ready, 0)

    def _start_monitor_when_ready(self, waited: int) -> None:
        if EMU_CONFIG.exists():
            launch("monitor.py", EMU_CONFIG, "--out", ROOT / "runs" / "monitor_emulation", "--open")
        elif waited < 180:
            self.after(2000, self._start_monitor_when_ready, waited + 2)
        else:
            messagebox.showerror("Эмуляция", "Эмуляторы не запустились — см. их окно консоли.")

    def service(self) -> None:
        if not SERVICE_CONFIG.exists():
            shutil.copy(ROOT / "config" / "printers.example.yaml", SERVICE_CONFIG)
            messagebox.showinfo("Настройка", "Впишите принтеры (адрес, серийный номер, код доступа) в открывшийся файл, "
                                             "сохраните и нажмите кнопку ещё раз.")
            subprocess.Popen(["notepad", str(SERVICE_CONFIG)])
            return
        if self.need_model():
            launch("monitor.py", SERVICE_CONFIG, "--open")

    # ---------------------------------------------------------------- модель и данные
    def report(self, name: str) -> None:
        html = latest(f"models/{name}/*.eval.html")
        if html:
            os.startfile(html)
        elif (ROOT / "models" / f"{name}.pt").exists():
            launch("evaluate.py", ROOT / "models" / f"{name}.pt", "--open")
        else:
            messagebox.showinfo("Нет модели", "Эта модель ещё не обучена.")

    def benchmark(self) -> None:
        if self.need_model():
            out = ROOT / "runs" / "benchmark" / f"{stamp()}_ui"
            launch("benchmark.py", "--synthetic", 3, "--out", out, open_after=out / "report.html")

    def train(self, name: str, base: str) -> None:
        if messagebox.askyesno("Обучение", f"Обучить модель «{name}» заново? Идёт на видеокарте, окно можно не трогать."):
            launch("train.py", name, "--model", base)

    def label_tool(self) -> None:
        if not any((ROOT / "data" / "labeled").glob("*/data.yaml")):
            messagebox.showinfo("Разметка", "Размеченных партий пока нет (data\\labeled).")
            return
        subprocess.Popen([str(PY.with_name("pythonw.exe")), str(ROOT / "label_tool.py")], cwd=ROOT)

    def prepare_labeling(self) -> None:
        out = next_batch(ROOT / "data" / "to_label")
        launch("extract_frames.py", "--out", out, "--prelabel", open_after=out)

    def tests(self) -> None:
        flags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        subprocess.Popen(["cmd", "/k", str(PY), "-m", "pytest", "-q"], cwd=ROOT, creationflags=flags)


def main() -> None:
    if not PY.exists():
        tk.Tk().withdraw()
        messagebox.showerror("Нет окружения", f"Не найден {PY}.\nЗапустите install.bat.")
        return
    if os.name == "nt":
        try:  # без этого на экранах с масштабом 125–150% окно размытое
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72)
    except tk.TclError:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
