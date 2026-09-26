"""Итоговая приёмка: логика тревог ПРИЛОЖЕНИЯ (guard.domain.policy) на сохранённых детекциях benchmark.py.

    python tools/acceptance.py runs/benchmark/onnx_train_synth runs/benchmark/onnx_holdout_synth

Модель не запускается — берутся detections.pkl, поэтому перебор порогов и правил подтверждения
занимает секунды. Порог выбирается на первом прогоне (записи из обучения), проверяется на втором
(отложенные печати) — так проверка честная. Критерий выбора: минимум ложных остановок, затем больше пойманных
настоящих сбоев, затем искусственных; при равенстве — порог выше. Запас: порог годится, только если
и ступенью ниже (на 0.1) ложных остановок не больше — модель на новых принтерах «поплывёт».

Метрики на 10 часов печати: ложные остановки (главное), ложные уведомления; пойманные сбои
(тревога любого рода в окне сбоя), из них остановкой печати; среднее время до тревоги.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import astuple, dataclass
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guard.domain.models import Detection  # noqa: E402
from guard.domain.policy import AlertPolicy, Settings  # noqa: E402

GRACE = timedelta(seconds=180)
RULES = [(3, 5), (4, 6), (5, 8)]          # (кадров с проблемой, из последних N)
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


@dataclass
class Score:
    hours: float = 0.0
    false_stops: int = 0
    false_notes: int = 0
    real: int = 0
    real_caught: int = 0
    synth: int = 0
    synth_caught: int = 0
    synth_stopped: int = 0
    delays: list = None

    def rate(self, n: int) -> float:
        return n / max(self.hours, 1e-9) * 10


def evaluate(run: dict, s: Settings) -> Score:
    sc = Score(delays=[])
    fails = run.get("failures", [])
    for seg in run["segments"].values():
        frames = seg["frames"]
        sc.hours += (frames[-1] - frames[0]).total_seconds() / 3600
        pol = AlertPolicy(s, has_scene=False)          # тревоги сцены (нет пластины и т.п.) здесь не считаются
        t0 = frames[0]
        alerts = []
        for ts, dets in seg["judged"]:
            v = pol.update([Detection(*d) for d in dets], (ts - t0).total_seconds())
            alerts += [(ts, c, c in v.stop_for) for c in v.fired]
        seg_fails = [f for f in fails if f[0] <= frames[-1] and f[1] >= frames[0]]
        for ts, c, stop in alerts:
            if not any(a <= ts <= b + GRACE for a, b, _ in seg_fails):
                sc.false_stops += stop
                sc.false_notes += not stop
        for a, b, note in seg_fails:
            hits = [(ts, stop) for ts, _, stop in alerts if a <= ts <= b + GRACE]
            synthetic = note.startswith("синтетика")
            if synthetic:
                sc.synth += 1
                sc.synth_caught += bool(hits)
                sc.synth_stopped += any(stop for _, stop in hits)
            else:
                sc.real += 1
                sc.real_caught += bool(hits)
            if hits:
                sc.delays.append((min(t for t, _ in hits) - a).total_seconds())
    return sc


def fuse(a: dict, b: dict, mode: str) -> dict:
    """Две модели брака на одних и тех же кадрах → одна «defects» (guard.domain.fusion):
    and — обе нашли на том же месте (уверенность по меньшей), or — любая, avg — средняя (нет у второй = 0)."""
    from guard.domain.fusion import fuse_defects
    out = {**a, "model": f"{Path(a['model']).name} {mode} {Path(b['model']).name}", "segments": {}}
    for name, sa in a["segments"].items():
        sb = b["segments"][name]
        assert sa["frames"] == sb["frames"], f"{name}: прогоны на разных кадрах"
        judged_b = dict(sb["judged"])
        seg = []
        for ts, da in sa["judged"]:
            db = judged_b.get(ts, [])
            da_def = [Detection(*d) for d in da if d[0] == "defects"]
            db_def = [Detection(*d) for d in db if d[0] == "defects"]
            fused = fuse_defects(da_def, db_def, mode)
            seg.append((ts, [d for d in da if d[0] != "defects"] + [astuple(d) for d in fused]))
        out["segments"][name] = {"frames": sa["frames"], "judged": seg}
    return out


def row(label: str, sc: Score) -> str:
    d = f"{sum(sc.delays) / len(sc.delays):.0f} с" if sc.delays else "—"
    return (f"| {label} | {sc.hours:.0f} | **{sc.rate(sc.false_stops):.2f}** ({sc.false_stops}) | "
            f"{sc.rate(sc.false_notes):.2f} ({sc.false_notes}) | {sc.real_caught}/{sc.real} | "
            f"{sc.synth_caught}/{sc.synth} ({sc.synth_stopped} остановкой) | {d} |")


HEAD = ("| настройка | часов | ложные остановки /10 ч | ложные уведомления /10 ч | настоящие сбои | "
        "искусственные сбои | время до тревоги |\n|---|---:|---:|---:|---|---|---:|")


def select_and_check(calib: dict, check: dict, thresholds: list[float]) -> tuple[list[str], tuple, Score]:
    """Сетка на calib, выбор с запасом, проверка выбранного на check."""
    lines = [f"# Приёмка: {Path(calib['model']).name}", "",
             f"Выбор порога — на «{calib['set']}», проверка — на «{check['set']}». Логика тревог — приложения "
             f"(останавливают только: {', '.join(Settings().stop_on)}).", "", f"## Выбор ({calib['set']})", "", HEAD]
    grid = []
    for hits, window in RULES:
        prev = None
        for t in thresholds:
            s = Settings(defect_conf=t, min_hits=hits, window=window)
            sc = evaluate(calib, s)
            if prev is not None:        # запас: и на пороге ступенью ниже ложных не больше
                fs = max(sc.false_stops, prev.false_stops)
                grid.append(((fs, -sc.real_caught, -sc.synth_stopped, sc.false_notes, -t), (t, hits, window), sc))
            prev = sc
            lines.append(row(f"{hits} из {window}, порог {t}", sc))
    best = min(grid, key=lambda g: g[0])
    t, hits, window = best[1]
    checked = evaluate(check, Settings(defect_conf=t, min_hits=hits, window=window))
    lines += ["", f"**Выбрано: {hits} из {window}, порог {t}.**", "", f"## Проверка ({check['set']})", "", HEAD,
              row(f"{hits} из {window}, порог {t}", checked)]
    return lines, (best[1], best[2]), checked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calib", type=Path, help="прогон для выбора порога (записи из обучения)")
    ap.add_argument("check", type=Path, help="прогон для проверки (отложенные печати)")
    ap.add_argument("--second", type=Path, nargs=2, metavar=("CALIB2", "CHECK2"),
                    help="прогоны второй модели на тех же кадрах — оценить пару моделей (and / or / avg)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    calib = pickle.load(open(args.calib / "detections.pkl", "rb"))
    check = pickle.load(open(args.check / "detections.pkl", "rb"))

    if not args.second:
        lines, *_ = select_and_check(calib, check, args.thresholds)
    else:
        calib2 = pickle.load(open(args.second[0] / "detections.pkl", "rb"))
        check2 = pickle.load(open(args.second[1] / "detections.pkl", "rb"))
        variants = [("только первая", calib, check), ("только вторая", calib2, check2)]
        variants += [(mode, fuse(calib, calib2, mode), fuse(check, check2, mode)) for mode in ("and", "or", "avg")]
        summary = [f"# Пара моделей: {Path(calib['model']).name} + {Path(calib2['model']).name}", "",
                   "Для каждого варианта порог и правило выбраны на «train» (с запасом), ниже — выбранное на «holdout».",
                   "", "| вариант | выбрано | train: ложные / настоящие / искусств. | " + HEAD.split("\n")[0].split(" | ", 1)[1],
                   "|---|---|---|" + HEAD.split("\n")[1][5:]]
        details = []
        for label, c1, c2 in variants:
            part, ((t, hits, window), train_sc), sc = select_and_check(c1, c2, args.thresholds)
            summary.append(f"| {label} | {hits} из {window}, {t} | {train_sc.false_stops} / {train_sc.real_caught}"
                           f"/{train_sc.real} / {train_sc.synth_stopped}/{train_sc.synth} " + row("", sc)[2:])
            details += ["", f"# {label}"] + part[1:]
        lines = summary + details
    md = "\n".join(lines) + "\n"
    print(md)
    if args.out:
        args.out.write_text(md, encoding="utf-8")
        print(f"→ {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
