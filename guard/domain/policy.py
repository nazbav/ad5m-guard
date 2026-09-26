"""Правило «когда останавливать печать». Чистая логика: детекции и время на входе, решения на выходе."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, fields

from guard.domain.models import Detection

NO_PLATE = "plate_not_visible"
SCENE_PROBLEMS = frozenset({"pei_misplaced", "head_no_cover", "glass_plate"})
PLATES = frozenset({"pei_plate", "glass_plate"})
ACTIONS = ("pause", "cancel", "none")


@dataclass
class Settings:
    """Настройки распознавания и реакции. Хранятся в конфиге, меняются из панели."""
    interval_s: float = 5.0
    action: str = "pause"                      # pause | cancel | none (только уведомить)
    # Калибровка ONNX-модели (tools/acceptance.py, 26.09.2026): на 72 ч записей и 24 ч отложенных
    # печатей 0 ложных остановок, и с запасом — на пороге 0.4 тоже 0. Отчёт: docs/acceptance.md
    defect_conf: float = 0.5
    scene_conf: float = 0.4
    window: int = 6
    min_hits: int = 4
    cooldown_s: float = 600
    hand_hold_s: float = 10
    # Подтверждение по месту: кадр считается попаданием, только если проблема найдена там же
    # (IoU ≥ spatial_iou), что и на одном из предыдущих кадров окна. Ложные срабатывания «прыгают»
    # по кадру, настоящее спагетти держится на месте и растёт.
    spatial: bool = True
    spatial_iou: float = 0.2
    check_plate: bool = True                   # тревога, если пластина PEI не видна
    # «Фон печати»: что модель находит в первые baseline_s секунд печати (тестовая полоска AD5M,
    # старые следы на пластине), дальше на том же месте не считается проблемой. Кроме stop_on —
    # сбой в первые минуты печати пропускать нельзя.
    baseline_s: float = 180
    # Главное — минимум ложных остановок: печать останавливают ТОЛЬКО эти классы,
    # остальные найденные проблемы — уведомление. Новый класс сам по себе ничего не остановит.
    stop_on: list[str] = field(default_factory=lambda: ["spaghetti", "detached"])
    # Тревога-уведомление без остановки. Всё, что не в stop_on и не в notify_on, только видно в панели.
    notify_on: list[str] = field(default_factory=lambda: ["plate_not_visible", "glass_plate"])
    muted: list[str] = field(default_factory=lambda: ["pei_misplaced"])      # не учитывать вовсе
    telegram_token: str = ""
    telegram_chat_id: str = ""
    keep_alert_images_days: int = 30
    # При тревоге сохранять исходные кадры (без рамок) — кадр тревоги и столько до него — в training/
    # для разметки и дообучения моделей. 0 — не сохранять.
    training_frames: int = 10
    calibration: str = ""                    # под какой набор моделей подобраны порог и правило

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f"Действие при тревоге: {ACTIONS}")
        if not 1 <= int(self.min_hits) <= int(self.window) <= 50:
            raise ValueError("Нужно 1 ≤ «из скольких» ≤ «окно» ≤ 50")
        for k in ("defect_conf", "scene_conf"):
            if not 0.05 <= float(getattr(self, k)) <= 0.99:
                raise ValueError(f"{k} должен быть от 0.05 до 0.99")
        if not 1 <= float(self.interval_s) <= 300:
            raise ValueError("Интервал опроса — от 1 до 300 секунд")
        if not 0 <= int(self.training_frames) <= 60:
            raise ValueError("Кадров для обучения — от 0 до 60")

    def to_dict(self, secrets: bool = True) -> dict:
        d = asdict(self)
        if not secrets and d["telegram_token"]:
            d["telegram_token"] = "••••"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        return cls(**{**RECOMMENDED, **{k: v for k, v in (d or {}).items() if k in known}})

    @classmethod
    def recommended(cls) -> "Settings":
        return cls(**RECOMMENDED)


# Порог и правило, откалиброванные для поставляемого набора моделей (models.yaml → recommended),
# поверх значений по умолчанию выше (они — калибровка одной модели YOLO). Заполняет guard.app при запуске.
RECOMMENDED: dict = {}


@dataclass(frozen=True)
class Verdict:
    problems: frozenset[str]        # что не так на этом кадре
    fired: tuple[str, ...]          # тревоги, которые пора поднять
    stop_for: tuple[str, ...]       # из них — требующие остановки печати
    judged: bool                    # кадр учитывался (не рука в кадре)
    ignored: tuple = ()             # находки «фона печати» — не проблема, на кадре не рисуются


def _inside(a, b) -> float:
    """Доля площади a внутри b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class AlertPolicy:
    """Состояние одной печати. Проблема поднимает тревогу, если встретилась в min_hits из
    последних window проверок, и не чаще cooldown_s. Пока в кадре рука и hand_hold_s после —
    кадры не судятся (человек чистит стол или снимает деталь)."""

    def __init__(self, s: Settings, has_scene: bool):
        self.s = s
        self.has_scene = has_scene
        self._hits: dict[str, deque[bool]] = {}
        self._boxes: dict[str, deque[list]] = {}
        self._last: dict[str, float] = {}
        self._hand_until = float("-inf")
        self._seen: float | None = None
        self._t0: float | None = None
        self._background: dict[str, list] = {}

    def _split_background(self, dets: list[Detection], t: float) -> tuple[list[Detection], list[Detection]]:
        """(что судить, фон печати). Первые baseline_s секунд находки не-stop_on классов запоминаются
        как фон; дальше на тех же местах они игнорируются, новые места — судятся как обычно."""
        kept, ignored = [], []
        learning = t - self._t0 < self.s.baseline_s              # 0 — правило выключено
        for d in dets:
            if d.model != "defects" or d.cls in self.s.stop_on:
                kept.append(d)
                continue
            places = self._background.setdefault(d.cls, [])
            if learning:
                if not any(_iou(d.box, b) >= 0.7 for b in places):
                    places.append(d.box)
                ignored.append(d)
            elif any(_iou(d.box, b) >= 0.3 or _inside(d.box, b) >= 0.6 for b in places):
                ignored.append(d)
            else:
                kept.append(d)
        return kept, ignored

    def problems(self, dets: list[Detection]) -> frozenset[str]:
        found = {d.cls for d in dets if d.model == "defects" and d.conf >= self.s.defect_conf}
        scene = {d.cls for d in dets if d.model == "scene" and d.conf >= self.s.scene_conf}
        found |= scene & SCENE_PROBLEMS
        if self.has_scene and self.s.check_plate and not scene & PLATES:
            found.add(NO_PLATE)
        return frozenset(found)

    def _consistent(self, cls: str, dets: list[Detection]) -> bool:
        """Есть ли рамка класса рядом с его рамкой на одном из предыдущих кадров окна.
        Проблемы без рамки (нет пластины) — всегда «на месте»."""
        boxes = [d.box for d in dets if d.cls == cls and d.conf >= (self.s.defect_conf if d.model == "defects" else self.s.scene_conf)]
        hist = self._boxes.setdefault(cls, deque(maxlen=max(1, self.s.window - 1)))
        if not boxes:
            return cls == NO_PLATE
        ok = any(_iou(b, p) >= self.s.spatial_iou for prev in hist for p in prev for b in boxes)
        hist.append(boxes)
        return ok

    def reset(self) -> None:
        """Забыть попадания (не кулдауны): после перерыва в наблюдении или вмешательства человека
        старые кадры уже не говорят о том, что на столе сейчас."""
        self._hits.clear()
        self._boxes.clear()

    def update(self, dets: list[Detection], t: float) -> Verdict:
        if self._seen is not None and t - self._seen > max(60.0, 4 * float(self.s.interval_s)):
            self.reset()                        # пластину не было видно, камера/принтер молчали
        self._seen = t
        if any(d.model == "scene" and d.cls == "hand" and d.conf >= self.s.scene_conf for d in dets):
            self._hand_until = t + self.s.hand_hold_s
            self.reset()
        if t < self._hand_until:
            return Verdict(frozenset(), (), (), False)
        if self._t0 is None:
            self._t0 = t
        dets, ignored = self._split_background(dets, t)
        found = self.problems(dets)
        judged = (found - set(self.s.muted)) & (set(self.s.stop_on) | set(self.s.notify_on))
        hits = {c: self._consistent(c, dets) for c in judged} if self.s.spatial else {c: True for c in judged}
        for cls in judged:
            self._hits.setdefault(cls, deque(maxlen=self.s.window))
        fired = []
        for cls, hist in self._hits.items():
            hist.append(hits.get(cls, False))
            # тревога — только если проблема видна на ЭТОМ кадре и подтверждена историей
            if not hist[-1] or sum(hist) < self.s.min_hits:
                continue
            last = self._last.get(cls)
            if last is not None and t - last < self.s.cooldown_s:
                continue
            self._last[cls] = t
            fired.append(cls)
        fired.sort()
        stop = tuple(c for c in fired if c in self.s.stop_on)
        return Verdict(found, tuple(fired), stop, True, tuple(ignored))
