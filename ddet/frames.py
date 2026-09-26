"""Кадры с камеры принтера: время съёмки из имени файла и деление на печати."""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Sequence, TypeVar

T = TypeVar("T")

# Так называет кадры грабер: printer_20250309_154916.jpg. Roboflow дописывает
# к имени хвост вида _jpg.rf.<hash>.jpg, поэтому ищем шаблон, а не всё имя.
TS_RE = re.compile(r"printer_(\d{8})_(\d{6})")


def parse_timestamp(name: str) -> datetime | None:
    m = TS_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def split_sessions(items: Sequence[tuple[datetime, T]], max_gap: timedelta) -> list[list[tuple[datetime, T]]]:
    """Делит кадры на печати: разрыв во времени больше max_gap начинает новую."""
    sessions: list[list[tuple[datetime, T]]] = []
    for item in sorted(items, key=lambda x: x[0]):
        if sessions and item[0] - sessions[-1][-1][0] <= max_gap:
            sessions[-1].append(item)
        else:
            sessions.append([item])
    return sessions
