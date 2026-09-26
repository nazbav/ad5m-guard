"""Приёмочный тест модели брака: ложные остановки на час печати и пойманные сбои.

    python benchmark.py                               # отложенные печати (честная оценка)
    python benchmark.py --set all                     # все записанные печати AD5M
    python benchmark.py --defects models/defects_rfdetr_small.pth --defect-conf 0.3

Каждая печать прогоняется как у сервиса: кадр раз в every_s секунд → модель брака →
правило тревоги приложения (guard.domain.policy): останавливают только классы stop_on.
Сбои и критерии — config/benchmark.yaml; отложенные печати — config/holdout.yaml.
Тревоги сцены (стекло, пластина) здесь не считаются: в записи нет статуса принтера.

Отчёт: runs/benchmark/<дата>/report.html и кадры каждой ложной остановки.
"""
from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ddet.config import CONFIG_DIR, ROOT
from ddet.detect import Detector, draw
from ddet.frames import split_sessions
from ddet.holdout import Holdout
from ddet.modelinfo import recommended_conf
from ddet.report import md_to_html, page
from ddet.synth import growing_event, load_library, plate_point
from guard.domain.policy import AlertPolicy, Settings

SYNTH_DURATION_S = 480


@dataclass
class Failure:
    session: str
    start: datetime
    end: datetime
    kind: str
    note: str
    detected_at: datetime | None = None


@dataclass
class SessionResult:
    name: str
    hours: float
    frames: int
    judged: int = 0
    stops: list[tuple[datetime, str]] = field(default_factory=list)
    false_stops: list[tuple[datetime, str]] = field(default_factory=list)


def load_config() -> dict:
    return yaml.safe_load((CONFIG_DIR / "benchmark.yaml").read_text(encoding="utf-8"))


def recorded_segments(which: str, holdout: Holdout) -> dict[str, list]:
    """Записанные печати AD5M: имя → [(время, чтение кадра)].

    holdout — отрезки config/holdout.yaml (печати, которых не видела ни одна модель);
    train — остальные записи без этих отрезков; all — всё."""
    from extract_frames import scan, thumbnail
    found = scan([ROOT.parent], exclude=[ROOT])
    frames = sorted((c.taken, c.read) for c in found.values())
    out: dict[str, list] = {}
    if which == "holdout":
        for a, b, _ in holdout.ranges:
            seg = [(ts, r) for ts, r in frames if a <= ts <= b]
            if seg:
                out[f"{a:%Y%m%d_%H%M}"] = seg
    else:
        pool = [(ts, r) for ts, r in frames if which == "all" or ts not in holdout]
        for s in split_sessions(pool, timedelta(minutes=20)):
            out[f"{s[0][0]:%Y%m%d_%H%M}"] = s
    return {k: v for k, v in out.items() if len(v) >= 20 and thumbnail(v[0][1]()) is not None}


def sample(frames: list, every_s: float) -> list:
    out, last = [], None
    for ts, read in frames:
        if last is None or (ts - last).total_seconds() >= every_s:
            out.append((ts, read))
            last = ts
    return out


def in_window(ts: datetime, failures: list[Failure], grace: timedelta) -> Failure | None:
    return next((f for f in failures if f.start <= ts <= f.end + grace), None)


def detect_session(frames: list, detector: Detector, scene_conf: float) -> list[tuple]:
    """Модель по кадрам печати один раз: [(время, чтение кадра, детекции)] только для кадров,
    где видна пластина — без неё принтер не печатает (снимают деталь, камера сдвинута), сервис бы не судил."""
    judged = []
    for i in range(0, len(frames), 16):
        chunk = frames[i:i + 16]
        images = [cv2.imdecode(np.frombuffer(read(), np.uint8), cv2.IMREAD_COLOR) for _, read in chunk]
        for (ts, read), dets in zip(chunk, detector.batch(images)):
            if "scene" in detector.models and not any(
                    d.model == "scene" and d.cls in ("pei_plate", "glass_plate") and d.conf >= scene_conf for d in dets):
                continue
            judged.append((ts, read, dets))
    return judged


def simulate(name: str, frames: list, judged: list[tuple], cfg: Settings, failures: list[Failure],
             grace: timedelta, out: Path | None) -> SessionResult:
    """Логика тревог сервиса по готовым детекциям при пороге cfg.defect_conf."""
    policy = AlertPolicy(cfg, has_scene=False)  # тревоги сцены здесь не считаются
    res = SessionResult(name, (frames[-1][0] - frames[0][0]).total_seconds() / 3600, len(frames), len(judged))
    t0 = frames[0][0]
    for ts, read, dets in judged:
        for cls in policy.update(dets, (ts - t0).total_seconds()).stop_for:
            res.stops.append((ts, cls))
            f = in_window(ts, failures, grace)
            if f is not None:
                f.detected_at = f.detected_at or ts
                continue
            res.false_stops.append((ts, cls))
            if out is not None:
                img = cv2.imdecode(np.frombuffer(read(), np.uint8), cv2.IMREAD_COLOR)
                shot = draw(img, dets, {"defects": cfg.defect_conf, "scene": cfg.scene_conf},
                            f"{ts:%Y-%m-%d %H:%M:%S}", alert=cls)
                (out / "false_stops").mkdir(parents=True, exist_ok=True)
                cv2.imencode(".jpg", shot)[1].tofile(str(out / "false_stops" / f"{ts:%Y%m%d_%H%M%S}_{cls}.jpg"))
    return res


def run_session(name: str, frames: list, detector: Detector, cfg: Settings, failures: list[Failure],
                grace: timedelta, out: Path | None) -> SessionResult:
    return simulate(name, frames, detect_session(frames, detector, cfg.scene_conf), cfg, failures, grace, out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defects", type=Path, default=ROOT / "models" / "defects.pt")
    ap.add_argument("--defect-conf", type=float, help="порог тревоги (по умолчанию — подобранный для модели)")
    ap.add_argument("--scene", type=Path, default=ROOT / "models" / "scene.pt",
                    help="модель сцены: кадры без видимой пластины не судим (как сервис вне печати)")
    ap.add_argument("--thresholds", type=float, nargs="*", default=[0.3, 0.4, 0.5, 0.6],
                    help="посчитать ещё и для этих порогов (кривая «ложные остановки / пойманные сбои»)")
    ap.add_argument("--synthetic", type=int, default=0, help="искусственных сбоев на каждую печать (из приёмочных вырезок)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", choices=("holdout", "train", "all"), default="holdout")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    bench = load_config()
    grace = timedelta(seconds=bench["grace_s"])
    conf = args.defect_conf if args.defect_conf is not None else recommended_conf(args.defects, Settings.defect_conf)
    cfg = Settings(defect_conf=conf)
    scene = args.scene if args.scene and args.scene.exists() else None
    detector = Detector(args.defects, scene, device=args.device, min_conf=min([0.2, conf, *args.thresholds]))
    out = args.out or ROOT / "runs" / "benchmark" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.defects.stem}_{args.set}"
    out.mkdir(parents=True, exist_ok=True)

    chosen = recorded_segments(args.set, Holdout.load())
    spans = {k: (v[0][0], v[-1][0]) for k, v in chosen.items()}
    failures = []
    for f in bench["failures"]:
        start, end = datetime.fromisoformat(f["start"]), datetime.fromisoformat(f["end"])
        seg = next((k for k, (a, b) in spans.items() if start <= b and end >= a), None)  # окна пересекаются
        if seg:
            failures.append(Failure(seg, start, end, f["kind"], f.get("note", "")))
    print(f"Модель {args.defects.name}, порог {conf}; отрезков печати {len(chosen)}, известных сбоев {len(failures)}")

    # Модель по кадрам — один раз; логика тревог — для каждого порога из --thresholds.
    extra = sorted({t for t in args.thresholds if t != conf})
    rng = np.random.default_rng(args.seed)
    cuts = load_library(ROOT / "data" / "synth" / "cutouts", "bench") if args.synthetic else []
    detected_frames = {}
    for name, frames in sorted(chosen.items()):
        frames = sample(frames, bench["every_s"])
        # Искусственные сбои: «растущий» комок из приёмочной половины вырезок в случайные моменты печати.
        n = len(frames)
        slots = np.linspace(0.1 * n, 0.8 * n, args.synthetic + 2)[1:-1].astype(int) if args.synthetic and n > 200 else []
        for idx in slots:
            idx = int(np.clip(idx + rng.integers(-20, 21), 0, n - 1))
            cut = cuts[rng.integers(len(cuts))].rgba
            # Пластина — по модели сцены на кадре начала сбоя: камеру между печатями переставляли.
            onset = cv2.imdecode(np.frombuffer(frames[idx][1](), np.uint8), cv2.IMREAD_COLOR)
            plates = [d for d in detector(onset) if d.model == "scene" and d.cls == "pei_plate" and d.conf >= 0.5]
            if not plates:
                continue
            center = plate_point(max(plates, key=lambda d: d.conf).box, rng)
            width = float(rng.uniform(70, 200))
            frames = growing_event(frames, idx, SYNTH_DURATION_S, cut, center, width)
            end = frames[idx][0] + timedelta(seconds=SYNTH_DURATION_S)
            failures.append(Failure(name, frames[idx][0], end, "spaghetti", "синтетика: растущий комок"))
        detected_frames[name] = (frames, detect_session(frames, detector, cfg.scene_conf))
        print(f"  {name}: кадров {len(frames)}, оценено {len(detected_frames[name][1])}")

    # Детекции — на диск: логику тревог можно перебирать без модели (tools/logic_lab.py).
    with open(out / "detections.pkl", "wb") as fh:
        pickle.dump({"model": str(args.defects), "set": args.set, "every_s": bench["every_s"],
                     "failures": [(f.start, f.end, f.note) for f in failures],
                     "segments": {n: {"frames": [ts for ts, _ in fr],
                                      "judged": [(ts, [dataclasses.astuple(d) for d in dets]) for ts, _, dets in j]}
                                  for n, (fr, j) in detected_frames.items()}}, fh)

    def evaluate(thr: float, out_dir: Path | None) -> tuple[list[SessionResult], list[Failure]]:
        fails = [dataclasses.replace(f) for f in failures]
        c = dataclasses.replace(cfg, defect_conf=thr)
        return [simulate(n, fr, j, c, [f for f in fails if f.session == n], grace, out_dir)
                for n, (fr, j) in detected_frames.items()], fails

    curve = []
    for thr in extra:
        rs, fs = evaluate(thr, None)
        h = sum(r.hours for r in rs)
        curve.append((thr, sum(len(r.false_stops) for r in rs) / max(h, 1e-9) * 10,
                      sum(1 for f in fs if f.detected_at), len(fs)))
    results, failures = evaluate(conf, out)

    hours = sum(r.hours for r in results)
    false = sum(len(r.false_stops) for r in results)
    fs_rate = false / max(hours, 1e-9) * 10
    detected = [f for f in failures if f.detected_at]
    det_rate = len(detected) / len(failures) if failures else None
    delays = [(f.detected_at - f.start).total_seconds() for f in detected]
    mean_delay = sum(delays) / len(delays) if delays else None
    c = bench["criteria"]
    checks = [("ложных остановок на 10 ч печати", f"{fs_rate:.2f}", f"≤ {c['max_false_stops_per_10h']}",
               fs_rate <= c["max_false_stops_per_10h"])]
    for title, group in (("пойманы настоящие сбои", [f for f in failures if not f.note.startswith("синтетика")]),
                         ("пойманы искусственные сбои", [f for f in failures if f.note.startswith("синтетика")])):
        if not group:
            continue
        caught = [f for f in group if f.detected_at]
        rate = len(caught) / len(group)
        checks.append((title, f"{len(caught)} из {len(group)} ({rate:.0%})", f"≥ {c['min_detection_rate']:.0%}",
                       rate >= c["min_detection_rate"]))
    if failures:
        checks.append(("среднее время до остановки", f"{mean_delay:.0f} с" if mean_delay is not None else "—",
                       f"≤ {c['max_time_to_detect_s']} с",
                       mean_delay is not None and mean_delay <= c["max_time_to_detect_s"]))
    passed = all(ok for *_, ok in checks)

    lines = [f"# Приёмочный тест: {args.defects.name} ({args.set})", "",
             f"- порог тревоги брака: {conf}; кадр раз в {bench['every_s']} с; тревога — {cfg.min_hits} из {cfg.window} проверок",
             f"- печатей: {len(results)}, часов печати: {hours:.1f}, известных сбоев: {len(failures)}",
             "", f"**Итог: {'ГОДНО' if passed else 'НЕ ГОДНО'}**", "",
             "| критерий | результат | требование | |", "|---|---|---|---|"]
    lines += [f"| {n} | {v} | {req} | {'✓' if ok else '✗'} |" for n, v, req, ok in checks]
    if curve:
        rows = sorted(curve + [(conf, fs_rate, len(detected), len(failures))])
        lines += ["", "## Другие пороги", "", "| порог | ложных на 10 ч | пойманы сбои |", "|---:|---:|---|"]
        lines += [f"| {t:.2f}{' ←' if t == conf else ''} | {r:.2f} | {d} из {n} |" for t, r, d, n in rows]
    if failures:
        lines += ["", "## Сбои", "", "| печать | окно | что | поймано | через |", "|---|---|---|---|---|"]
        for f in failures:
            d = f"{(f.detected_at - f.start).total_seconds():.0f} с" if f.detected_at else "—"
            lines.append(f"| {f.session} | {f.start:%H:%M}–{f.end:%H:%M} | {f.note} | "
                         f"{'да' if f.detected_at else '**нет**'} | {d} |")
    lines += ["", "## По печатям", "", "Оценено — кадры, где видна пластина (принтер может печатать).", "",
              "| печать | часов | кадров / оценено | остановок | ложных | ложные (время — класс) |",
              "|---|---:|---:|---:|---:|---|"]
    for r in results:
        fs = ", ".join(f"{t:%H:%M} {cls}" for t, cls in r.false_stops) or "—"
        lines.append(f"| {r.name} | {r.hours:.1f} | {r.frames} / {r.judged} | {len(r.stops)} | {len(r.false_stops)} | {fs} |")
    if false:
        lines += ["", f"Кадры ложных остановок — папка `false_stops` ({false} шт.)."]
    md = "\n".join(lines) + "\n"
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "report.html").write_text(page("Приёмочный тест", md_to_html(md)), encoding="utf-8")
    print(md)
    print(f"Отчёт: {out / 'report.html'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
