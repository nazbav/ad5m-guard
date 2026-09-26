"""Сложные отрицательные примеры: кадры нормальной печати, где модель ложно видит сбой.

    python tools/hard_negatives.py runs/benchmark/v2_train_synth --out data/labeled/hardneg1

Берёт детекции прогона benchmark.py (только записи из обучения — отложенные печати не трогаются),
выбирает кадры вне окон известных сбоев, где модель дала класс из stop_on с уверенностью ≥ --conf,
и кладёт исходные кадры (без вклеек) с пустой разметкой. Набор помечается «только для модели брака»:
в модель сцены он не попадает (там пустая разметка означала бы «пластины нет»).
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from datetime import timedelta
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.config import ROOT, labeling_names  # noqa: E402
from ddet.holdout import Holdout  # noqa: E402
from guard.domain.policy import Settings  # noqa: E402

MARGIN = timedelta(minutes=10)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "labeled" / "hardneg1")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--min-gap-s", type=float, default=30, help="не брать кадры чаще (соседние почти одинаковы)")
    args = ap.parse_args(argv)
    run = pickle.load(open(args.run / "detections.pkl", "rb"))
    if run["set"] != "train":
        raise SystemExit("Сложные примеры берутся только из записей обучения (--set train), иначе проверка нечестная")
    stop = set(Settings().stop_on)
    fails = run.get("failures", [])
    holdout = Holdout.load()
    picked = []
    for seg in run["segments"].values():
        last = None
        for ts, dets in seg["judged"]:
            if ts in holdout or any(a - MARGIN <= ts <= b + MARGIN for a, b, _ in fails):
                continue
            if not any(d[0] == "defects" and d[1] in stop and d[2] >= args.conf for d in dets):
                continue
            if last and (ts - last).total_seconds() < args.min_gap_s:
                continue
            picked.append(ts)
            last = ts
    from extract_frames import scan
    frames = {c.taken: c for c in scan([ROOT.parent], exclude=[ROOT]).values()}
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True)
    (args.out / "labels").mkdir()
    n = 0
    for ts in picked:
        c = frames.get(ts)
        if c is None:
            continue
        (args.out / "images" / c.name).write_bytes(c.read())
        (args.out / "labels" / (Path(c.name).stem + ".txt")).write_text("", encoding="utf-8")
        n += 1
    names = labeling_names()
    (args.out / "data.yaml").write_text(yaml.safe_dump({"train": "images", "val": "images", "nc": len(names), "names": names,
                                                        "only_for": "defects"}, allow_unicode=True, sort_keys=False),
                                        encoding="utf-8")
    print(f"Сложных отрицательных кадров: {n} → {args.out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
