"""Наблюдение за одним принтером: из детекций очередного кадра решает, когда тревога.

Это ядро будущего сервиса на много принтеров — по экземпляру на принтер.
"""
from __future__ import annotations

from dataclasses import dataclass

from ddet.alerts import AlertTracker
from ddet.detect import Detection

# Классы сцены, которые сами по себе — проблема.
SCENE_PROBLEMS = {"pei_misplaced", "head_no_cover", "glass_plate"}
# Модель сцены не видит пластину: её нет, камеру сдвинули или закрыли —
# в любом случае печать сейчас без присмотра.
NO_PLATE = "plate_not_visible"


@dataclass
class WatchConfig:
    defect_conf: float = 0.4
    scene_conf: float = 0.4
    window: int = 5            # сколько последних проверок смотреть
    min_hits: int = 3          # сколько из них должны показать проблему
    cooldown_s: float = 600    # не повторять ту же тревогу чаще
    hand_hold_s: float = 10    # после руки в кадре столько секунд не судим
    # Замечаются и попадают в отчёты, но тревогу не поднимают. pei_misplaced —
    # пока модель его не выучила (24 примера): на записях срабатывает почти в каждой печати.
    muted: frozenset[str] = frozenset({"pei_misplaced"})
    # Тревога — только уведомление, печать не останавливается: мусор на столе
    # и струны посреди печати обычно не повод её губить.
    notify_only: frozenset[str] = frozenset({"garbage", "stringing"})


class PrintWatcher:
    def __init__(self, cfg: WatchConfig, has_scene: bool):
        self.cfg = cfg
        self.has_scene = has_scene
        self.tracker = AlertTracker(cfg.window, cfg.min_hits, cfg.cooldown_s)
        self.hand_until = float("-inf")

    def problems(self, dets: list[Detection]) -> set[str]:
        """Что не так на этом кадре (без учёта истории)."""
        found = {d.cls for d in dets if d.model == "defects" and d.conf >= self.cfg.defect_conf}
        scene = {d.cls for d in dets if d.model == "scene" and d.conf >= self.cfg.scene_conf}
        found |= scene & SCENE_PROBLEMS
        if self.has_scene and not scene & {"pei_plate", "glass_plate"}:
            found.add(NO_PLATE)
        return found

    def update(self, dets: list[Detection], t: float) -> tuple[set[str], list[str]]:
        """Возвращает (проблемы на кадре, тревоги, которые пора поднять).

        Пока в кадре рука и ещё hand_hold_s секунд после — кадры не учитываются:
        человек чистит стол или снимает модель, это не брак."""
        if any(d.model == "scene" and d.cls == "hand" and d.conf >= self.cfg.scene_conf for d in dets):
            self.hand_until = t + self.cfg.hand_hold_s
        if t < self.hand_until:
            return set(), []
        found = self.problems(dets)
        return found, self.tracker.update(found - self.cfg.muted, t)
