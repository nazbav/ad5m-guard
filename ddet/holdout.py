"""Отложенные печати (config/holdout.yaml): интервалы времени, кадры которых — только test."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from ddet.config import CONFIG_DIR

HOLDOUT_YAML = CONFIG_DIR / "holdout.yaml"


class Holdout:
    def __init__(self, ranges: list[tuple[datetime, datetime, str]]):
        self.ranges = ranges

    @classmethod
    def load(cls, path: Path | None = HOLDOUT_YAML) -> "Holdout":
        if path is None or not Path(path).exists():
            return cls([])
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls([(datetime.fromisoformat(r["from"]), datetime.fromisoformat(r["to"]), r.get("note", ""))
                    for r in data.get("ranges", [])])

    def __bool__(self) -> bool:
        return bool(self.ranges)

    def __contains__(self, ts: datetime | None) -> bool:
        return ts is not None and any(a <= ts <= b for a, b, _ in self.ranges)
