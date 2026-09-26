"""Хранилища: конфиг в YAML (атомарная запись), журнал событий в SQLite, снимки тревог — файлами."""
from __future__ import annotations

import os
import json
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import yaml

from guard.domain.models import Event, PrinterConfig
from guard.domain.policy import Settings


class YamlConfigStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> tuple[Settings, list[PrinterConfig]]:
        if not self.path.exists():
            return Settings.recommended(), []
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        return Settings.from_dict(data.get("settings") or {}), [PrinterConfig.from_dict(p) for p in data.get("printers") or []]

    def save(self, settings: Settings, printers: list[PrinterConfig]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = ("# AD5M Guard — настройки. Удобнее менять в панели: Настройки и «Добавить принтер».\n"
                + yaml.safe_dump({"settings": settings.to_dict(), "printers": [p.to_dict() for p in printers]},
                                 allow_unicode=True, sort_keys=False))
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".config-", suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, self.path)           # атомарно: при сбое питания конфиг не будет полупустым


class SqliteEventStore:
    def __init__(self, db: Path, images: Path, keep_days: int = 30):
        self.db, self.images, self.keep_days = Path(db), Path(images), keep_days
        self.images.mkdir(parents=True, exist_ok=True)
        self.training = self.images.parent / "training"            # исходные кадры тревог для дообучения
        self.db.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, printer_id TEXT, printer TEXT,
            kind TEXT, detail TEXT, image TEXT)""")
        self._conn.execute("CREATE INDEX IF NOT EXISTS ev_printer ON events(printer_id, id)")
        self._conn.commit()
        self.cleanup()

    def add(self, event: Event, jpg: bytes | None) -> Event:
        if jpg:
            name = f"{event.when.replace(':', '-').replace(' ', '_')}_{event.printer_id}_{event.kind}.jpg"
            (self.images / name).write_bytes(jpg)
            event.image = name
        with self._lock:
            self._conn.execute("INSERT INTO events(ts, printer_id, printer, kind, detail, image) VALUES (?,?,?,?,?,?)",
                               (event.when, event.printer_id, event.printer, event.kind, event.detail, event.image))
            self._conn.commit()
        return event

    def recent(self, limit: int = 100, printer_id: str | None = None) -> list[Event]:
        q = "SELECT ts, printer_id, printer, kind, detail, image FROM events"
        args: tuple = ()
        if printer_id:
            q += " WHERE printer_id = ?"
            args = (printer_id,)
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY id DESC LIMIT ?", args + (int(limit),)).fetchall()
        return [Event(*r) for r in rows]

    def image(self, name: str) -> bytes | None:
        p = (self.images / name).resolve()
        if p.parent != self.images.resolve() or not p.exists():   # защита от ../
            return None
        return p.read_bytes()

    def save_sample(self, label: str, jpgs: list[bytes], meta: dict) -> Path:
        """Исходные кадры тревоги для дообучения: training/<время>_<принтер>_<класс>/frame_-NN.jpg + meta.json."""
        safe = "".join(c if c.isalnum() or c in "-_+" else "_" for c in label)[:80]
        folder = self.training / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{safe}"
        folder.mkdir(parents=True, exist_ok=True)
        for k, jpg in enumerate(jpgs):
            last = k == len(jpgs) - 1                              # по порядку времени, последний — тревога
            (folder / f"{k:02d}{'_alert' if last else ''}.jpg").write_bytes(jpg)
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return folder

    def training_samples(self, days: int = 30) -> list[Path]:
        cutoff = time.time() - days * 86400
        return sorted(d for d in self.training.glob("*") if d.is_dir() and d.stat().st_mtime >= cutoff)

    def export_training(self, dst: Path, days: int = 30, info: dict | None = None) -> dict:
        """Один zip для передачи разработчикам на дообучение: папки тревог за days дней + info.json."""
        import zipfile
        samples = self.training_samples(days)
        summary = {**(info or {}), "days": days, "alerts": len(samples),
                   "from": samples[0].name[:10] if samples else None, "to": samples[-1].name[:10] if samples else None}
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:          # JPEG уже сжаты
            z.writestr("info.json", json.dumps(summary, ensure_ascii=False, indent=1))
            for d in samples:
                for f in sorted(d.iterdir()):
                    z.write(f, f"{d.name}/{f.name}")
        return summary

    def cleanup(self) -> None:
        cutoff = time.time() - self.keep_days * 86400
        for p in self.images.glob("*.jpg"):
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
        for d in self.training.glob("*"):
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
