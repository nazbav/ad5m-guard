"""Подобрать порог тревоги модели брака на валидации и записать его рядом с весами.

    python tools/dump_predictions.py yolo models/defects.pt --split val --out runs/calib/defects_val.json
    python tools/calibrate.py runs/calib/defects_val.json --weights models/defects.pt

Порог — наибольшая доля замеченных проблемных кадров AD5M при доле ложных тревог
на чистых кадрах AD5M не выше --max-false-alarm. Пишется в <веса>.json
({"defect_conf": ...}); run_video.py и monitor.py берут его, если порог не задан явно.
Подбирать на val, а не на test: иначе проверка на test будет завышена.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ddet.metrics import frame_level  # noqa: E402


def pick_threshold(truth: dict, preds: dict, problems: set[int], max_fa: float) -> tuple[float, dict]:
    best = (0.95, frame_level(truth, preds, problems, 0.95))
    for thr in np.arange(0.95, 0.049, -0.01):
        r = frame_level(truth, preds, problems, float(thr))
        if r["false_alarm"] <= max_fa:
            best = (round(float(thr), 2), r)
        else:
            break
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", type=Path, help="предсказания на val (tools/dump_predictions.py --split val)")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--max-false-alarm", type=float, default=0.05)
    args = ap.parse_args(argv)

    from compare import load_truth
    m = json.loads(args.preds.read_text(encoding="utf-8"))
    files = sorted(m["images"])
    truth = load_truth(Path(m["data"]), m["split"], files, m["names"])
    own = {f: v for f, v in truth.items() if f.startswith("own_")} or truth
    problems = set(range(len(m["names"])))
    thr, r = pick_threshold(own, m["images"], problems, args.max_false_alarm)
    side = args.weights.with_suffix(".json")
    data = json.loads(side.read_text(encoding="utf-8")) if side.exists() else {}
    data.update({"defect_conf": thr, "calibration": {
        "split": m["split"], "frames": len(own), "max_false_alarm": args.max_false_alarm,
        "recall": round(r["recall"], 3), "false_alarm": round(r["false_alarm"], 3)}})
    side.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Порог {thr}: на кадрах AD5M ({m['split']}) замечено {r['recall']:.0%} проблемных, "
          f"ложных тревог {r['false_alarm']:.0%} → {side}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
