"""Прогон моделей по записям печати: что видит модель и в какой момент остановила бы печать.

    python run_video.py ..\\ff5\\printer_timelapse.mp4
    python run_video.py ..\\ff5\\printer_20250418_121536.zip --every 10
    python run_video.py --all --every 10          # все записанные печати AD5M из родительской папки проекта
    python run_video.py "http://192.168.0.108:8080/?action=stream" --every 2   # живая камера, Ctrl+C — стоп

Источник: видеофайл; папка или zip с кадрами грабера (printer_ГГГГММДД_ччммсс.jpg) —
они сами делятся на отдельные печати; URL потока камеры. Модель сцены
подключается сама, если есть models/scene.pt.

На каждую печать — папка в runs/video/<партия>/:
  annotated.mp4  — видео с рамками; с момента тревоги кадр в красной рамке
  detections.csv — все детекции
  timeline.png   — какие проблемы и когда; вертикальные линии — тревоги
  summary.md     — когда поднялась бы каждая тревога
и общая таблица runs/video/<партия>/index.md.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from ddet.config import ROOT
from ddet.detect import Detector, draw
from ddet.frames import parse_timestamp, split_sessions
from ddet.modelinfo import recommended_conf
from ddet.report import md_to_html, page
from ddet.sources import (Frame, frames_from_entries, is_sequence, is_stream, open_source,
                          sequence_entries)
from guard.domain.policy import NO_PLATE, SCENE_PROBLEMS, AlertPolicy, Settings

CAMERA_SIZE = (640, 480)
SCENE_ALERTS = SCENE_PROBLEMS | {NO_PLATE}


@dataclass
class Job:
    name: str
    title: str
    frames: Iterator[Frame]
    seen_in_training: int = 0     # сколько кадров этой печати есть в размеченном датасете


@dataclass
class Result:
    job: Job
    out: Path
    frames: int
    duration_s: float
    alerts: list[tuple[float, str, str]] = field(default_factory=list)
    problem_frames: Counter = field(default_factory=Counter)


def safe_name(s: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", s)


def to_h264(path: Path) -> None:
    """OpenCV пишет MPEG-4 Part 2, его не везде открывают — перекодируем в H.264, если есть ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return
    tmp = path.with_name(path.stem + "_h264.mp4")
    r = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(path), "-c:v", "libx264", "-crf", "26",
                        "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)])
    if r.returncode == 0:
        tmp.replace(path)


def plot_timeline(rows: list[dict], alerts: list[tuple[float, str, str]], out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    problems = sorted({r["problem"] for r in rows if r["problem"]}) or ["(ничего не найдено)"]
    y = {p: i for i, p in enumerate(problems)}
    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.45 * len(problems)))
    pts = [(r["t"] / 60, y[r["problem"]], r["conf"]) for r in rows if r["problem"]]
    if pts:
        xs, ys, cs = zip(*pts)
        ax.scatter(xs, ys, c=cs, cmap="viridis", vmin=0, vmax=1, s=30, marker="|")
    for t, _, cls in alerts:
        ax.axvline(t / 60, color="red", lw=1)
        ax.text(t / 60, len(problems) - 0.4, f" {cls}", color="red", fontsize=8, va="top")
    ax.set_yticks(range(len(problems)), problems)
    ax.set_xlabel("минуты от начала")
    ax.set_ylim(-0.6, len(problems) - 0.2)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def run(job: Job, detector: Detector, cfg: Settings, out: Path, fps_out: float,
        max_frames: int | None) -> Result | None:
    watcher = AlertPolicy(cfg, has_scene="scene" in detector.models)     # правило тревоги приложения
    thresholds = {"defects": cfg.defect_conf, "scene": cfg.scene_conf}
    res = Result(job, out, 0, 0.0)
    rows: list[dict] = []
    writer = size = first_alert = None
    started = time.monotonic()
    video_path = out / "annotated.mp4"
    try:
        for fr in job.frames:
            if writer is None:
                out.mkdir(parents=True, exist_ok=True)
            dets = detector(fr.image)
            v = watcher.update(dets, fr.t)
            found, fired = v.problems, v.fired
            for cls in fired:
                res.alerts.append((fr.t, fr.label, cls))
                print(f"  {'ОСТАНОВКА' if cls in cfg.stop_on else 'УВЕДОМЛЕНИЕ'} {cls} — {fr.label}")
            stops = list(v.stop_for)
            first_alert = first_alert or (stops[0] if stops else None)
            res.problem_frames.update(found)
            for d in dets:
                rows.append({"t": round(fr.t, 2), "frame": fr.index, "label": fr.label, "model": d.model,
                             "class": d.cls, "conf": round(d.conf, 3),
                             "problem": d.cls if d.conf >= thresholds[d.model] and d.cls in found else "",
                             "x1": round(d.box[0]), "y1": round(d.box[1]), "x2": round(d.box[2]), "y2": round(d.box[3])})
            if NO_PLATE in found:
                rows.append({"t": round(fr.t, 2), "frame": fr.index, "label": fr.label, "model": "scene",
                             "class": NO_PLATE, "conf": 1.0, "problem": NO_PLATE,
                             "x1": "", "y1": "", "x2": "", "y2": ""})

            header = f"{fr.label}  " + (" ".join(sorted(found)) if found else "ok")
            img = draw(fr.image, dets, thresholds, header, alert=first_alert)
            for cls in fired:  # кадр каждой тревоги — отдельной картинкой, чтобы проверить глазами
                (out / "alerts").mkdir(exist_ok=True)
                shot = draw(fr.image, dets, thresholds, header, alert=cls,
                            kind="STOP" if cls in cfg.stop_on else "NOTE")
                name = f"{safe_name(fr.label)}_{cls}.jpg"
                (out / "alerts" / name).write_bytes(cv2.imencode(".jpg", shot)[1].tobytes())
            if writer is None:
                size = (img.shape[1], img.shape[0])
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, size)
            writer.write(img if (img.shape[1], img.shape[0]) == size else cv2.resize(img, size))
            res.frames += 1
            res.duration_s = fr.t
            if res.frames % 500 == 0:
                print(f"  {res.frames} кадров, {fr.label}, {res.frames / (time.monotonic() - started):.1f} кадр/с")
            if max_frames and res.frames >= max_frames:
                break
    except KeyboardInterrupt:
        print("  Остановлено вручную")
    finally:
        if writer is not None:
            writer.release()

    if res.frames == 0:
        print("  кадров нет — пропуск")
        return None
    to_h264(video_path)

    fields = ["t", "frame", "label", "model", "class", "conf", "problem", "x1", "y1", "x2", "y2"]
    with open(out / "detections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    plot_timeline(rows, res.alerts, out / "timeline.png", f"{job.title}  ({res.frames} кадров)")

    lines = [f"# {job.title}", "", f"- кадров обработано: {res.frames}",
             "- модели: " + ", ".join(f"{k}={Path(m.ckpt_path).name}" for k, m in detector.models.items()),
             f"- пороги: брак {cfg.defect_conf}, сцена {cfg.scene_conf}; "
             f"тревога — {cfg.min_hits} из {cfg.window} последних проверок"]
    if job.seen_in_training:
        lines.append(f"- ВНИМАНИЕ: {job.seen_in_training} кадров этой печати были в обучении — оценка завышена")
    lines.append("")
    stops = [a for a in res.alerts if a[2] in cfg.stop_on]
    if stops:
        lines.append(f"**Печать была бы остановлена: {stops[0][1]} — {stops[0][2]}.**")
    else:
        lines.append("Печать не была бы остановлена.")
    if res.alerts:
        lines += ["", "| когда | тревога | действие |", "|---|---|---|"]
        lines += [f"| {label} | {cls} | {'остановка' if cls in cfg.stop_on else 'уведомление'} |"
                  for _, label, cls in res.alerts]
    lines += ["", "Кадров, на которых модель видела проблему:", ""]
    lines += [f"- {cls}: {k} ({k / res.frames:.0%})" for cls, k in res.problem_frames.most_common()] or ["- ни одного"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_summary_html(out, job.title, "\n".join(lines))
    first = f"остановка {stops[0][1]} — {stops[0][2]}" if stops else "без остановки"
    print(f"  {res.frames} кадров → {first}, тревог всего {len(res.alerts)}")
    return res


def write_summary_html(out: Path, title: str, md: str) -> None:
    """Страница печати: отчёт, таймлайн, видео и кадры тревог."""
    parts = [md_to_html(md), "<h2>Таймлайн</h2>", '<img src="timeline.png" alt="таймлайн">',
             "<h2>Видео</h2>", '<video src="annotated.mp4" controls preload="metadata"></video>']
    shots = sorted((out / "alerts").glob("*.jpg")) if (out / "alerts").exists() else []
    if shots:
        parts.append("<h2>Кадры тревог</h2><div class=gallery>")
        parts += [f'<figure><a href="alerts/{p.name}"><img src="alerts/{p.name}" loading=lazy></a>'
                  f"<figcaption>{p.stem}</figcaption></figure>" for p in shots]
        parts.append("</div>")
    (out / "summary.html").write_text(page(title, "\n".join(parts)), encoding="utf-8")


# ------------------------------------------------------------------ задания
def labeled_timestamps() -> set[str]:
    from extract_frames import labeled_timestamps as lt
    return lt(ROOT / "data" / "raw")


def is_camera_frame(read) -> bool:
    img = cv2.imdecode(np.frombuffer(read(), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    return img is not None and (img.shape[1] * 8, img.shape[0] * 8) == CAMERA_SIZE


def session_jobs(entries: list[tuple[str, callable]], every: float, gap_min: float, min_frames: int,
                 labeled: set[str], prefix: str = "") -> list[Job]:
    stamped = [(parse_timestamp(Path(n).name), (n, r)) for n, r in entries]
    if not stamped or any(ts is None for ts, _ in stamped):
        return [Job(prefix or "frames", prefix or "frames", frames_from_entries(entries, every))]
    jobs = []
    for s in split_sessions(stamped, timedelta(minutes=gap_min)):
        start, end = s[0][0], s[-1][0]
        if len(s) < min_frames:
            continue
        if not is_camera_frame(s[0][1][1]):
            print(f"  {start:%Y-%m-%d %H:%M}: не камера AD5M — пропуск")
            continue
        seen = sum(1 for ts, (n, _) in s if f"printer_{ts:%Y%m%d_%H%M%S}" in labeled)
        name = f"{start:%Y%m%d_%H%M}"
        title = f"Печать {start:%Y-%m-%d %H:%M}–{end:%H:%M} ({(end - start).total_seconds() / 3600:.1f} ч)"
        jobs.append(Job(name, title, frames_from_entries([e for _, e in s], every), seen))
    return jobs


def write_index(results: list[Result], out: Path, stop_on: list[str]) -> None:
    """Остановки из-за брака, уведомления и тревоги сцены — в разных колонках: в записи
    нет статуса принтера, и «пластина не видна» после печати (пластину сняли) — это не брак."""
    lines = ["# Прогон по записям печати", "",
             "«В обучении» — были ли кадры этой печати в размеченном датасете (тогда оценка завышена).",
             "Тревоги сцены в записи без статуса принтера бывают и после конца печати — смотрите время.", "",
             "| печать | кадров | в обучении | остановка из-за брака | уведомления | тревоги сцены | кадры с проблемой |",
             "|---|---:|:---:|---|---|---|---|"]
    for r in results:
        stops = [(label, c) for _, label, c in r.alerts if c in stop_on]
        notes = sorted({c for _, _, c in r.alerts if c not in stop_on and c not in SCENE_ALERTS})
        scene = sorted({c for _, _, c in r.alerts if c in SCENE_ALERTS})
        first = f"{stops[0][0]} — **{stops[0][1]}**" if stops else "—"
        probs = ", ".join(f"{c} {k / r.frames:.0%}" for c, k in r.problem_frames.most_common(3)) or "—"
        seen = "да" if r.job.seen_in_training else "нет"
        link = f"[{r.job.title}]({r.out.name}/summary.md)"
        lines.append(f"| {link} | {r.frames} | {seen} | {first} | {', '.join(notes) or '—'} | "
                     f"{', '.join(scene) or '—'} | {probs} |")
    md = "\n".join(lines) + "\n"
    (out / "index.md").write_text(md, encoding="utf-8")
    (out / "index.html").write_text(page("Прогон по записям печати", md_to_html(md.replace("/summary.md)", "/summary.html)"))),
                                    encoding="utf-8")
    print(f"\nСводка: {out / 'index.html'}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--all", action="store_true", help="все записанные печати из родительской папки проекта (кроме этого проекта)")
    ap.add_argument("--only", help="только эти печати, через запятую, как в сводке: 20250304_2057,20250305_1249")
    ap.add_argument("--defects", type=Path, default=ROOT / "models" / "defects.pt")
    ap.add_argument("--scene", type=Path, default=ROOT / "models" / "scene.pt", help="модель сцены (если файла нет — без неё)")
    ap.add_argument("--no-scene", action="store_true")
    ap.add_argument("--every", type=float, default=0, help="брать кадр не чаще чем раз в столько секунд (0 — все)")
    ap.add_argument("--defect-conf", type=float, help="порог тревоги брака (по умолчанию — подобранный для модели, иначе 0.4)")
    ap.add_argument("--scene-conf", type=float, default=Settings.scene_conf)
    ap.add_argument("--window", type=int, default=Settings.window)
    ap.add_argument("--min-hits", type=int, default=Settings.min_hits)
    ap.add_argument("--session-gap-min", type=float, default=20)
    ap.add_argument("--min-frames", type=int, default=20, help="печати короче пропускать")
    ap.add_argument("--fps-out", type=float, default=10, help="кадров в секунду в итоговом видео")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--out", type=Path, help="папка партии (по умолчанию runs/video/<дата_время>)")
    ap.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--open", action="store_true", help="открыть сводку в браузере по окончании")
    args = ap.parse_args(argv)
    if not args.sources and not args.all:
        ap.error("укажите источник или --all")

    if not args.defects.exists():
        raise SystemExit(f"Нет модели брака {args.defects}. Обучите: python train.py defects")
    scene = None if args.no_scene or not args.scene.exists() else args.scene
    detector = Detector(args.defects, scene, device=args.device)
    defect_conf = args.defect_conf if args.defect_conf is not None else recommended_conf(args.defects, Settings.defect_conf)
    cfg = Settings(defect_conf=defect_conf, scene_conf=args.scene_conf, window=args.window, min_hits=args.min_hits)
    out_root = args.out or ROOT / "runs" / "video" / time.strftime("%Y%m%d_%H%M%S")
    labeled = labeled_timestamps()

    jobs: list[Job] = []
    if args.all:
        from extract_frames import scan
        found = scan([ROOT.parent], exclude=[ROOT])
        jobs += session_jobs([(c.name, c.read) for c in found.values()], args.every, args.session_gap_min,
                             args.min_frames, labeled)
    for src in args.sources:
        if is_sequence(src):
            jobs += session_jobs(sequence_entries(src), args.every, args.session_gap_min, args.min_frames,
                                 labeled, prefix=safe_name(Path(src).stem))
        else:
            name = "live_" + time.strftime("%Y%m%d_%H%M%S") if is_stream(src) else safe_name(Path(src).stem)
            jobs.append(Job(name, src, open_source(src, args.every)))

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        jobs = [j for j in jobs if j.name in wanted]
    results = []
    for job in jobs:
        print(f"\n=== {job.title}")
        r = run(job, detector, cfg, out_root / job.name, args.fps_out, args.max_frames)
        if r:
            results.append(r)
    if results:
        write_index(results, out_root, cfg.stop_on)
        if args.open and hasattr(os, "startfile"):
            os.startfile(out_root / "index.html")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
