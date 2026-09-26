"""Сведения о весах модели, которые хранятся рядом с ними: <веса>.json (порог тревоги и т.п.)."""
from __future__ import annotations

import json
from pathlib import Path


def recommended_conf(weights: Path, default: float) -> float:
    """Порог тревоги, подобранный tools/calibrate.py для этих весов; иначе default."""
    side = Path(weights).with_suffix(".json")
    try:
        return float(json.loads(side.read_text(encoding="utf-8"))["defect_conf"])
    except (OSError, ValueError, KeyError):
        return default
