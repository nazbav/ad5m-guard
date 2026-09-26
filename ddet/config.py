"""Загрузка конфигурации классов и общие пути проекта."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DEFECTS_YAML = CONFIG_DIR / "defects.yaml"


@dataclass(frozen=True)
class ClassConfig:
    name: str                          # имя модели = имя файла конфига (defects, scene)
    names: list[str]
    roboflow_map: dict[str, str | None]
    only_own: bool                     # учить только на кадрах своей камеры


def load_classes(path: Path = DEFECTS_YAML) -> ClassConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = list(data["names"])
    mapping = dict(data.get("roboflow_map") or {})
    unknown = {v for v in mapping.values() if v is not None} - set(names)
    if unknown:
        raise ValueError(f"В roboflow_map есть классы, которых нет в names: {sorted(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError("В names есть повторы")
    return ClassConfig(path.stem, names, mapping, bool(data.get("only_own", False)))


def labeling_names() -> list[str]:
    """Классы для новой разметки кадров: брак и сцена в одном проекте разметки."""
    return load_classes(CONFIG_DIR / "defects.yaml").names + load_classes(CONFIG_DIR / "scene.yaml").names


def long_path(p: Path | str) -> str:
    """Windows не открывает пути длиннее 260 символов без префикса \\\\?\\."""
    s = str(Path(p).resolve())
    if os.name == "nt" and len(s) >= 240 and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s
