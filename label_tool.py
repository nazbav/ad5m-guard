"""Просмотр и правка разметки кадров (папка в формате YOLO: images/, labels/, data.yaml).

    python label_tool.py                       # последняя папка из data/labeled
    python label_tool.py data/labeled/batch1

Управление:
    ← →  / PgUp PgDn    предыдущий / следующий кадр
    мышь: тянуть        новая рамка текущего класса (выбран справа)
    клик по рамке       выделить;  Delete — удалить;  1–9, 0 — сменить класс выделенной
    S                   показать/скрыть стол, голову, тестовую линию (классы сцены)
Изменения сохраняются сами при переходе к другому кадру и при закрытии.
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import yaml
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
CONTEXT = {"pei_plate", "printer_head", "test_line"}   # «фоновые» классы сцены
COLORS = ["#e53935", "#fb8c00", "#fdd835", "#43a047", "#00acc1", "#8e24aa", "#3949ab", "#6d4c41",
          "#d81b60", "#00897b", "#7cb342", "#5e35b1", "#f4511e"]


class Box:
    def __init__(self, cls: int, x1: float, y1: float, x2: float, y2: float):
        self.cls, self.x1, self.y1, self.x2, self.y2 = cls, x1, y1, x2, y2

    @classmethod
    def parse(cls, line: str, w: int, h: int) -> "Box":
        c, xc, yc, bw, bh = line.split()[:5]
        xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
        return cls(int(c), xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2)

    def line(self, w: int, h: int) -> str:
        x1, x2 = sorted((max(0, self.x1), min(w, self.x2)))
        y1, y2 = sorted((max(0, self.y1), min(h, self.y2)))
        return f"{self.cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


class Labeler(ttk.Frame):
    SCALE = 1.5

    def __init__(self, master: tk.Tk, folder: Path):
        super().__init__(master, padding=6)
        self.folder = folder
        data = yaml.safe_load((folder / "data.yaml").read_text(encoding="utf-8"))
        names = data["names"]
        self.names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
        self.all_images = sorted((folder / "images").glob("*.jpg"))
        self.images = self.all_images
        self.i = 0
        self.boxes: list[Box] = []
        self.selected: Box | None = None
        self.dirty = False
        self.drag = None
        self.show_context = tk.BooleanVar(value=False)
        self.only_defects = tk.BooleanVar(value=False)
        self.current = tk.IntVar(value=0)

        master.title(f"Разметка — {folder}")
        self.grid(sticky="nsew")
        self.canvas = tk.Canvas(self, width=int(640 * self.SCALE), height=int(480 * self.SCALE), bg="black",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, rowspan=2)
        side = ttk.Frame(self, padding=(8, 0))
        side.grid(row=0, column=1, sticky="n")
        ttk.Label(side, text="Класс новой рамки (клавиши 1–0):").pack(anchor="w")
        for k, n in enumerate(self.names):
            key = str((k + 1) % 10) if k < 10 else " "
            ttk.Radiobutton(side, text=f"{key}  {n}", value=k, variable=self.current,
                            command=self.recolor_selected).pack(anchor="w")
        ttk.Separator(side).pack(fill="x", pady=6)
        ttk.Checkbutton(side, text="Показывать стол/голову/линию (S)", variable=self.show_context,
                        command=self.redraw).pack(anchor="w")
        ttk.Checkbutton(side, text="Только кадры с браком", variable=self.only_defects,
                        command=self.apply_filter).pack(anchor="w")
        ttk.Separator(side).pack(fill="x", pady=6)
        nav = ttk.Frame(side)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀", width=4, command=lambda: self.go(-1)).pack(side="left")
        ttk.Button(nav, text="▶", width=4, command=lambda: self.go(1)).pack(side="left", padx=4)
        ttk.Button(nav, text="Удалить рамку", command=self.delete).pack(side="left")
        self.status = ttk.Label(side, justify="left", wraplength=260)
        self.status.pack(anchor="w", pady=8)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        for key, fn in (("<Left>", lambda e: self.go(-1)), ("<Right>", lambda e: self.go(1)),
                        ("<Prior>", lambda e: self.go(-1)), ("<Next>", lambda e: self.go(1)),
                        ("<Delete>", lambda e: self.delete()), ("<s>", lambda e: self.toggle_context()),
                        ("<S>", lambda e: self.toggle_context())):
            master.bind(key, fn)
        for d in range(10):
            master.bind(str(d), lambda e, d=d: self.pick((d - 1) % 10))
        master.protocol("WM_DELETE_WINDOW", self.close)
        self.load()

    # ---------------------------------------------------------------- данные
    def label_path(self, img: Path) -> Path:
        return self.folder / "labels" / f"{img.stem}.txt"

    def read_boxes(self, img: Path) -> list[Box]:
        p = self.label_path(img)
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        return [Box.parse(l, 640, 480) for l in text.splitlines() if l.strip()]

    def has_defect(self, img: Path) -> bool:
        return any(self.names[b.cls] not in CONTEXT | {"hand"} for b in self.read_boxes(img))

    def load(self) -> None:
        if not self.images:
            self.status.config(text="Нет кадров")
            return
        img_path = self.images[self.i]
        self.pil = Image.open(img_path).convert("RGB")
        self.w, self.h = self.pil.size
        self.photo = ImageTk.PhotoImage(self.pil.resize((int(self.w * self.SCALE), int(self.h * self.SCALE))))
        self.boxes = [Box.parse(l, self.w, self.h) for l in self._text(img_path).splitlines() if l.strip()]
        self.selected = None
        self.dirty = False
        self.redraw()

    def _text(self, img: Path) -> str:
        p = self.label_path(img)
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def save(self) -> None:
        if self.dirty and self.images:
            lines = [b.line(self.w, self.h) for b in self.boxes if abs(b.x2 - b.x1) > 2 and abs(b.y2 - b.y1) > 2]
            self.label_path(self.images[self.i]).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            self.dirty = False

    # ---------------------------------------------------------------- вид
    def visible(self, b: Box) -> bool:
        return self.show_context.get() or self.names[b.cls] not in CONTEXT

    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, image=self.photo, anchor="nw")
        s = self.SCALE
        for b in self.boxes:
            if not self.visible(b):
                continue
            color = COLORS[b.cls % len(COLORS)]
            width = 4 if b is self.selected else 2
            c.create_rectangle(b.x1 * s, b.y1 * s, b.x2 * s, b.y2 * s, outline=color, width=width)
            c.create_text(b.x1 * s + 2, b.y1 * s - 2, text=self.names[b.cls], anchor="sw", fill=color,
                          font=("Segoe UI", 10, "bold"))
        counts: dict[str, int] = {}
        for b in self.boxes:
            counts[self.names[b.cls]] = counts.get(self.names[b.cls], 0) + 1
        name = self.images[self.i].name if self.images else ""
        self.status.config(text=f"Кадр {self.i + 1} из {len(self.images)}\n{name}\n\n" +
                                "\n".join(f"{k}: {v}" for k, v in sorted(counts.items())))

    def toggle_context(self) -> None:
        self.show_context.set(not self.show_context.get())
        self.redraw()

    def apply_filter(self) -> None:
        self.save()
        current = self.images[self.i] if self.images else None
        self.images = [p for p in self.all_images if self.has_defect(p)] if self.only_defects.get() else self.all_images
        self.i = self.images.index(current) if current in self.images else 0
        self.load()

    # ---------------------------------------------------------------- действия
    def go(self, step: int) -> None:
        self.save()
        if self.images:
            self.i = max(0, min(len(self.images) - 1, self.i + step))
            self.load()

    def hit(self, x: float, y: float) -> Box | None:
        inside = [b for b in self.boxes if self.visible(b) and b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2]
        return min(inside, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1)) if inside else None

    def on_press(self, e) -> None:
        x, y = e.x / self.SCALE, e.y / self.SCALE
        box = self.hit(x, y)
        if box:
            self.selected = box
            self.current.set(box.cls)
            self.drag = None
        else:
            self.selected = None
            self.drag = (x, y, self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="white", dash=(3, 2)))
        self.redraw() if box else None

    def on_drag(self, e) -> None:
        if self.drag:
            x0, y0, rid = self.drag
            self.canvas.coords(rid, x0 * self.SCALE, y0 * self.SCALE, e.x, e.y)

    def on_release(self, e) -> None:
        if not self.drag:
            return
        x0, y0, _ = self.drag
        x1, y1 = e.x / self.SCALE, e.y / self.SCALE
        self.drag = None
        if abs(x1 - x0) > 3 and abs(y1 - y0) > 3:
            box = Box(self.current.get(), min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self.boxes.append(box)
            self.selected = box
            self.dirty = True
        self.redraw()

    def delete(self) -> None:
        if self.selected in self.boxes:
            self.boxes.remove(self.selected)
            self.selected = None
            self.dirty = True
            self.redraw()

    def pick(self, k: int) -> None:
        if k < len(self.names):
            self.current.set(k)
            self.recolor_selected()

    def recolor_selected(self) -> None:
        if self.selected:
            self.selected.cls = self.current.get()
            self.dirty = True
            self.redraw()

    def close(self) -> None:
        self.save()
        self.master.destroy()


def main() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        sets = sorted((ROOT / "data" / "labeled").glob("*/data.yaml"), key=lambda p: p.stat().st_mtime)
        folder = sets[-1].parent if sets else None
        if folder is None:
            root.withdraw()
            chosen = filedialog.askdirectory(title="Папка с разметкой (images, labels, data.yaml)")
            if not chosen:
                return
            folder = Path(chosen)
            root.deiconify()
    Labeler(root, folder)
    root.mainloop()


if __name__ == "__main__":
    main()
