"""Стенд логики тревог: разные правила «когда останавливать печать» на сохранённых детекциях приёмки.

    python tools/logic_lab.py runs/benchmark/lab_yolo_holdout runs/benchmark/lab_yolo_train

Модель не запускается — берутся detections.pkl из прогонов benchmark.py, поэтому любое
правило проверяется за секунды. Для каждого правила: ложные остановки на 10 ч печати
(по каждому прогону отдельно) и сколько известных сбоев поймано и через сколько.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ddet.config import CONFIG_DIR  # noqa: E402

STOP = {"spaghetti", "warping", "cracks", "detached"}   # остальное — уведомления
COOLDOWN = 600


@dataclass
class Rule:
    name: str
    kind: str                 # hits | ewm | ewm_base
    thr: float                # порог уверенности кадра (hits) или сглаженного счёта (ewm)
    window: int = 5
    min_hits: int = 3
    span: int = 12
    warmup: int = 30          # ewm_base: первые кадры печати — база, не судим

    def run(self, judged: list) -> list[tuple[datetime, str]]:
        stops, last = [], {}
        if self.kind == "hits":
            hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))
            for ts, dets in judged:
                found = {d[1] for d in dets if d[0] == "defects" and d[2] >= self.thr and d[1] in STOP}
                for c in STOP:
                    hist[c].append(c in found)
                    if sum(hist[c]) >= self.min_hits and (c not in last or (ts - last[c]).total_seconds() >= COOLDOWN):
                        stops.append((ts, c))
                        last[c] = ts
            return stops
        alpha = 2 / (self.span + 1)
        ewm = defaultdict(float)
        total = defaultdict(float)
        for k, (ts, dets) in enumerate(judged):
            for c in STOP:
                score = max((d[2] for d in dets if d[0] == "defects" and d[1] == c), default=0.0)
                ewm[c] = alpha * score + (1 - alpha) * ewm[c]
                total[c] += score
                if self.kind == "ewm":
                    value = ewm[c]
                else:
                    if k < self.warmup:
                        continue
                    value = ewm[c] - total[c] / (k + 1)     # выше обычного для этой печати
                if value >= self.thr and (c not in last or (ts - last[c]).total_seconds() >= COOLDOWN):
                    stops.append((ts, c))
                    last[c] = ts
        return stops


RULES = (
    [Rule(f"3 из 5, порог {t}", "hits", t) for t in (0.4, 0.5, 0.6, 0.7)]
    + [Rule(f"4 из 6, порог {t}", "hits", t, 6, 4) for t in (0.4, 0.5, 0.6)]
    + [Rule(f"6 из 10, порог {t}", "hits", t, 10, 6) for t in (0.4, 0.5, 0.6)]
    + [Rule(f"сглаживание, порог {t}", "ewm", t) for t in (0.3, 0.4, 0.5)]
    + [Rule(f"выше обычного, порог {t}", "ewm_base", t) for t in (0.2, 0.3, 0.4)]
)


def load_failures() -> list[tuple[datetime, datetime, str]]:
    data = yaml.safe_load((CONFIG_DIR / "benchmark.yaml").read_text(encoding="utf-8"))
    return [(datetime.fromisoformat(f["start"]), datetime.fromisoformat(f["end"]), f.get("note", ""))
            for f in data["failures"]], data["grace_s"]


def evaluate(rule: Rule, runs: list[dict], failures, grace_s: int) -> dict:
    grace = timedelta(seconds=grace_s)
    out = {"false": {}, "caught": 0, "total": 0, "delays": []}
    for run in runs:
        failures = run.get("failures", failures)   # прогон с искусственными сбоями хранит свой список
        false, hours = 0, 0.0
        for seg in run["segments"].values():
            frames = seg["frames"]
            hours += (frames[-1] - frames[0]).total_seconds() / 3600
            stops = rule.run(seg["judged"])
            fails = [f for f in failures if f[0] <= frames[-1] and f[1] >= frames[0]]
            for ts, _ in stops:
                if not any(a <= ts <= b + grace for a, b, _ in fails):
                    false += 1
            for a, b, _ in fails:
                out["total"] += 1
                hit = [ts for ts, _ in stops if a <= ts <= b + grace]
                if hit:
                    out["caught"] += 1
                    out["delays"].append((min(hit) - a).total_seconds())
        out["false"][run["set"]] = false / max(hours, 1e-9) * 10
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="папки прогонов benchmark.py (с detections.pkl)")
    args = ap.parse_args(argv)
    runs = [pickle.load(open(r / "detections.pkl", "rb")) for r in args.runs]
    failures, grace_s = load_failures()
    sets = [r["set"] for r in runs]
    print("| правило | " + " | ".join(f"ложных/10 ч ({s})" for s in sets) + " | поймано | среднее время |")
    print("|---|" + "---:|" * len(sets) + "---|---:|")
    for rule in RULES:
        r = evaluate(rule, runs, failures, grace_s)
        delay = f"{sum(r['delays']) / len(r['delays']):.0f} с" if r["delays"] else "—"
        print(f"| {rule.name} | " + " | ".join(f"{r['false'][s]:.2f}" for s in sets) +
              f" | {r['caught']} из {r['total']} | {delay} |")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
